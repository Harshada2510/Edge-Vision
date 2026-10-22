"""Exercise pinned NCNN weights, video timestamps and the executable runner."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from capture import VideoFile
from detector import FastestDet, decode_fastestdet
from lane_model import TriLiteLane, lane_lines
from main import arguments, run
from tracker import VehicleTracker

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def test_model_and_mask_on_cpu(self):
        detector = FastestDet(ROOT / "models", threads=1)
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        detections, metrics = detector(blank)
        self.assertIsInstance(detections, list)
        self.assertGreater(metrics["inference_ms"], 0)
        self.assertNotIn("torch", sys.modules)
        lane = TriLiteLane(ROOT / "models", threads=1)
        self.assertEqual(set(lane(blank)), {"left_line", "right_line"})
        self.assertEqual(lane(blank, include_mask=True)["mask"].shape, blank.shape[:2])

    def test_decoder_suppresses_own_bonnet(self):
        output = np.zeros((85, 22, 22), np.float32)
        output[0, 17, 10] = 0.99
        output[7, 17, 10] = 0.99  # COCO car
        output[3, 17, 10] = 2.5  # wide clipped bottom
        output[4, 17, 10] = 3
        self.assertEqual(decode_fastestdet(output, 640, 360), [])

    def test_tracker_cannot_return_unobserved_prediction(self):
        tr = VehicleTracker(min_hits=2)
        item = dict(class_name="car", confidence=0.9, x1=250, y1=150, x2=310, y2=240)
        first = tr.update([item], 0.0)
        self.assertFalse(first[0]["confirmed"])
        self.assertEqual(tr.update([], 0.1), [])
        second = tr.update([item], 0.2)
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertFalse(second[0]["confirmed"])  # streak broken by miss

    def test_lane_lines_require_both_boundaries(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        cv2.line(mask, (170, 275), (265, 185), 1, 5)
        self.assertIsNone(lane_lines(mask)["left_line"])
        cv2.line(mask, (470, 275), (375, 185), 1, 5)
        self.assertIsNotNone(lane_lines(mask)["left_line"])

    def test_real_runner_file_pts_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video, report = folder / "test.avi", folder / "report.json"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 360))
            self.assertTrue(writer.isOpened())
            for _ in range(18):
                writer.write(np.zeros((360, 640, 3), np.uint8))
            writer.release()
            src = VideoFile(video, loop=True)
            self.assertAlmostEqual(src.read()[1], 0.0)
            self.assertAlmostEqual(src.read()[1], 0.05)
            for _ in range(16):
                src.read()
            self.assertEqual(src.read()[3], 1)  # EOF loop generation
            src.close()
            result = run(arguments(["--video", str(video), "--no-web", "--threads", "1",
                                    "--max-frames", "18", "--warmup", "3", "--metrics", str(report)]))
            self.assertEqual(result["timed_frames"], 15)
            self.assertIsNotNone(result["observed_fps"])
            self.assertIn("frame_total_ms", json.loads(report.read_text())["stages"])


if __name__ == "__main__":
    unittest.main()
