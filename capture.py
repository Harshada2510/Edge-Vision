"""Camera latest-frame slot or video-file capture with source timestamps."""
import threading
import time

import cv2


class VideoFile:
    def __init__(self, path, loop=False, realtime=False):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not 0 < self.fps < 240:
            raise ValueError("Cannot establish video presentation timestamps (invalid FPS)")
        self.loop, self.realtime = loop, realtime
        self.index = 0
        self.wall_start = None
        self.generation = 0
        self.name = str(path)

    def read(self):
        if self.realtime and self.wall_start is not None:
            # Discard frames whose presentation deadline passed during inference.
            due = int((time.monotonic() - self.wall_start) * self.fps)
            while self.index < due:
                if not self.cap.grab():
                    break
                self.index += 1
        started = time.perf_counter()
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError("Video seek/loop failed")
            self.index, self.wall_start = 0, None
            self.generation += 1
            started = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Video is empty or cannot be replayed")
        if self.wall_start is None:
            self.wall_start = time.monotonic() - self.index / self.fps
        if self.realtime:
            wait = self.wall_start + self.index / self.fps - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        # frame index/FPS is deterministic even when OpenCV POS_MSEC is absent.
        pts = self.index / self.fps
        self.index += 1
        return frame, pts, (time.perf_counter() - started) * 1000, self.generation, 0.0

    def close(self):
        self.cap.release()


class LatestCamera:
    """Camera read thread overwrites a single slot; no growing frame queue."""
    def __init__(self, device=0, width=640, height=360, camera=None):
        self.cap = camera if camera is not None else cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")
        if camera is None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._condition = threading.Condition()
        self._packet, self._seq, self._closed = None, 0, False
        self.generation = 0
        self.name = "CSI CAM" if camera is not None else f"CAM {device}"
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        failures = 0
        while not self._closed:
            start = time.perf_counter()
            ok, frame = self.cap.read()
            received = time.monotonic()
            if not ok:
                failures += 1
                if failures >= 20:
                    with self._condition:
                        self._closed = True
                        self._condition.notify_all()
                    return
                time.sleep(0.05)
                continue
            if failures:
                self.generation += 1
            failures = 0
            with self._condition:
                self._seq += 1
                self._packet = (frame, received, (time.perf_counter() - start) * 1000,
                                self.generation, received)
                self._condition.notify_all()

    def read(self, timeout=2.0):
        with self._condition:
            previous = getattr(self, "_last_seq", 0)
            self._condition.wait_for(lambda: self._seq != previous or self._closed, timeout)
            if self._seq == previous:
                return None
            self._last_seq = self._seq
            return self._packet

    def close(self):
        self._closed = True
        self.cap.release()
        self._thread.join(timeout=2)


class _PicameraCapture:
    """Tiny Picamera2 adapter; RGB888 arrays are in OpenCV BGR byte order."""

    def __init__(self, width, height):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Install python3-picamera2 from Raspberry Pi OS for --picamera") from exc
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2, queue=False)
        self.camera.configure(config)
        self.camera.start()

    def isOpened(self):
        return True

    def read(self):
        frame = self.camera.capture_array()
        return frame is not None, frame

    def release(self):
        self.camera.stop()
        self.camera.close()


class LatestPiCamera(LatestCamera):
    """Libcamera CSI source with the same single-slot backpressure as USB."""

    def __init__(self, width=640, height=360):
        super().__init__(width=width, height=height,
                         camera=_PicameraCapture(width, height))
