"""EdgeVision Pi 3B runtime: capture -> NCNN -> tracking -> ADAS -> HUD.

Examples: python main.py --video clip.mp4 --no-web --max-frames 200
          python main.py --camera 0 --calibration config/my_camera.json
"""
import argparse
from collections import defaultdict, deque
import json
import platform
from pathlib import Path
import resource
import statistics
import threading
import time

TIMING_WINDOW_FRAMES = 1200  # bounded history for an always-on 1 GB Pi service
BUILD_ID = "pi3b-lane-distance-20260927-r4"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=f"EdgeVision {BUILD_ID}")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    source.add_argument("--picamera", action="store_true", help="CSI camera through Picamera2/libcamera")
    p.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    p.add_argument("--threads", type=int, default=3, choices=(1, 2, 3, 4))
    p.add_argument("--process-width", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--bonnet-bottom", type=float, default=0.90,
                   help="Suppress wide own-vehicle boxes ending below this frame fraction")
    p.add_argument("--calibration", type=Path)
    p.add_argument("--ego-speed-file", type=Path, help="Fresh external {unix_time,speed_mps} JSON for live FCDW")
    p.add_argument("--lanes", choices=("off", "trilite", "hough"), default="off")
    p.add_argument("--lane-overlay", action="store_true",
                   help="Visual TriLite test: raw mask, fit region and accepted lane pair")
    p.add_argument("--hud-style", choices=("compact", "full", "off"), default="compact",
                   help="OpenCV car-display overlay; off is for profiling only")
    p.add_argument("--lane-period", type=float, default=0.35)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--realtime", action="store_true", help="Pace file video; skip old frames if processing lags")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--stream-fps", type=float, default=10,
                   help="Maximum MJPEG encoding rate; inference and alerts still run on each frame")
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--metrics", type=Path, help="Write a full-pipeline timing report after completion")
    p.add_argument("--save-frame", type=Path, help="Save the last annotated frame for visual inspection")
    return p.parse_args(argv)


def describe(values):
    if not values:
        return None
    values = sorted(values)
    return {"median_ms": statistics.median(values),
            "p95_ms": values[min(len(values) - 1, int((len(values) - 1) * 0.95))],
            "mean_ms": statistics.mean(values)}


def resident_mib():
    """Read the process's current resident memory without extra packages."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def run(args):
    import cv2
    from capture import LatestCamera, LatestPiCamera, VideoFile
    from detector import FastestDet
    from pipeline import ADASPipeline
    from tracker import VehicleTracker
    from vehicle_state import EgoSpeedFile
    from web_server import TelemetryHub, create_cockpit_app

    if args.process_width < 320 or args.process_width > 1920:
        raise ValueError("--process-width must be 320–1920")
    if not 0.75 <= args.bonnet_bottom <= 1.0:
        raise ValueError("--bonnet-bottom must be between 0.75 and 1.0")
    if args.lane_period <= 0 or args.warmup < 0 or args.max_frames < 0:
        raise ValueError("Invalid cadence/frame count")
    if args.lane_overlay and args.lanes != "trilite":
        raise ValueError("--lane-overlay requires --lanes trilite")
    if not 0 < args.stream_fps <= 30 or not 40 <= args.jpeg_quality <= 95:
        raise ValueError("--stream-fps must be greater than 0 and at most 30; --jpeg-quality must be 40–95")
    cv2.setNumThreads(1)
    detector = FastestDet(args.model_dir, args.threads, args.threshold,
                          bonnet_bottom_frac=args.bonnet_bottom)
    lane_net = None
    if args.lanes == "trilite":
        from lane_model import TriLiteLane
        lane_net = TriLiteLane(args.model_dir, args.threads)
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    speed = EgoSpeedFile(args.ego_speed_file) if args.ego_speed_file else None
    source = (VideoFile(args.video, args.loop, args.realtime) if args.video else
              LatestPiCamera() if args.picamera else LatestCamera(args.camera))
    hub = TelemetryHub(hud_style=args.hud_style) if not args.no_web else None
    pipeline = None
    tracker = VehicleTracker()
    samples = defaultdict(lambda: deque(maxlen=TIMING_WINDOW_FRAMES))
    completion = deque(maxlen=TIMING_WINDOW_FRAMES)
    n = 0
    generation = None
    last_frame = None
    processing = {"frame": None}
    last_publish = float("-inf")
    last_rss = resident_mib()
    try:
        if hub is not None:
            app = create_cockpit_app(hub)
            threading.Thread(target=app.run,
                             kwargs={"host": args.host, "port": args.port,
                                     "threaded": True, "use_reloader": False},
                             daemon=True).start()
            if args.host == "0.0.0.0":
                print(f"Cockpit listening on port {args.port}; open "
                      f"http://YOUR_PI_HOSTNAME:{args.port}/ from the laptop", flush=True)
            else:
                print(f"Cockpit: http://{args.host}:{args.port}/", flush=True)
        while not args.max_frames or n < args.max_frames:
            started = time.perf_counter()
            packet = source.read()
            if packet is None:
                break
            raw, pts, capture_ms, current_generation, received = packet
            if generation != current_generation:
                if pipeline is not None:
                    pipeline.reset()
                tracker.reset()
                generation = current_generation
            h, w = raw.shape[:2]
            out_w = min(w, args.process_width)
            out_h = round(h * out_w / w)
            frame = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_AREA) if out_w != w else raw
            if pipeline is None:
                pipeline = ADASPipeline(out_w, out_h, enable_lane_detection=args.lanes != "off",
                                        ipm_config=calibration, lane_period_s=args.lane_period,
                                        bonnet_bottom_frac=args.bonnet_bottom,
                                        hud_style=args.hud_style, lane_overlay=args.lane_overlay)
                if lane_net is not None:
                    pipeline.lane_source = lambda now: lane_net(
                        processing["frame"], include_mask=args.lane_overlay)
                print(f"IPM: {pipeline.fcw.ipm.reason}; lanes: {args.lanes}; "
                      f"FCDW speed: {'external file' if speed else 'unavailable'}", flush=True)
            elif (out_w, out_h) != (pipeline.frame_w, pipeline.frame_h):
                raise ValueError("Frame dimensions changed during capture; restart and recalibrate")
            processing["frame"] = frame
            resized = time.perf_counter()
            if hub is not None:
                pipeline.apply_settings(hub.get_settings())
                pipeline.set_hud_layers(hub.get_hud_layers())
            detections, detector_times = detector(frame)
            track_started = time.perf_counter()
            tracks = tracker.update(detections, pts)
            tracked = time.perf_counter()
            stopped = speed.stopped() if speed and args.video is None else None
            tel = pipeline.process_frame(frame, tracks, detector_times["inference_ms"], pts,
                                         camera_source=(Path(source.name).name[:15] if args.video else source.name),
                                         ego_stopped=stopped)
            rendered = time.perf_counter()
            jpeg_started = time.perf_counter()
            jpeg_encoded = False
            if hub is not None and jpeg_started - last_publish >= 1 / args.stream_fps:
                hub.publish_frame(frame, jpeg_quality=args.jpeg_quality)
                last_publish = jpeg_started
                jpeg_encoded = True
            completed = time.perf_counter()
            tel["frame_total_ms"] = (completed - started) * 1000
            tel["frame_age_ms"] = ((time.monotonic() - received) * 1000 if received else None)
            if n % 20 == 0:
                last_rss = resident_mib()
            tel["rss_mib"] = last_rss
            # Telemetry publishes once per frame, after all time measurements.
            if hub is not None:
                hub.publish_telemetry(tel)
            n += 1
            last_frame = frame
            if n > args.warmup:
                sample = {**detector_times, "capture_ms": capture_ms,
                          "resize_ms": (resized - (started + capture_ms / 1000)) * 1000,
                          "tracker_ms": (tracked - track_started) * 1000,
                          "adas_hud_ms": (rendered - tracked) * 1000,
                          "pipeline_ms": tel["pipeline_ms"],
                          "hud_render_ms": tel["hud_render_ms"],
                          "jpeg_publish_ms": (completed - jpeg_started) * 1000,
                          "frame_total_ms": (completed - started) * 1000}
                for key, value in sample.items():
                    samples[key].append(max(0, value))
                if jpeg_encoded:
                    samples["jpeg_encode_ms"].append(max(0, (completed - jpeg_started) * 1000))
                completion.append(completed)
            if n % 60 == 0:
                print(f"frames={n} completed_fps={pipeline.current_fps or 0:.1f} "
                      f"detector_ms={detector_times['detector_ms']:.1f} "
                      f"end_to_end_ms={tel['frame_total_ms']:.1f}", flush=True)
    finally:
        source.close()
    if args.save_frame and last_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save_frame), last_frame):
            raise RuntimeError("Could not write annotated image")
    observed_fps = ((len(completion) - 1) / (completion[-1] - completion[0])
                    if len(completion) >= 2 and completion[-1] > completion[0] else None)
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "board": Path("/proc/device-tree/model").read_bytes().strip(b"\0").decode(errors="replace")
              if Path("/proc/device-tree/model").exists() else None,
              "source": source.name, "mode": "file" if args.video else "live_latest_slot",
              "file_realtime": args.realtime, "processed_size": [pipeline.frame_w, pipeline.frame_h] if pipeline else None,
              "threads": args.threads, "lanes": args.lanes, "hud_style": args.hud_style,
              "calibrated": pipeline.fcw.ipm.quality == "measured" if pipeline else False,
              "ipm_quality": pipeline.fcw.ipm.quality if pipeline else "unavailable",
              "rss_mib": resident_mib(),
              "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
              "web_enabled": hub is not None,
              "jpeg_frames_encoded": len(samples["jpeg_encode_ms"]),
              "frames": n, "warmup_excluded": min(n, args.warmup),
              "timed_frames": max(0, n - args.warmup),
              "timing_window_frames": len(completion), "observed_fps": observed_fps,
              "stages": {key: describe(values) for key, values in samples.items()},
              "note": "Full frame includes source wait/read, resize, NCNN, tracking, ADAS, HUD, and JPEG when web enabled. Camera read overlaps processing in a separate single-slot thread. For runs above 1200 timed frames, stage statistics and observed_fps cover only the latest 1200 frames. No Pi FPS claim unless board identifies the target Pi."}
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(arguments())
