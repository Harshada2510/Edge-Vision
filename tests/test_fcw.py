"""Fresh detections, plausible TTC and measured corridor tests."""
import sys
import math
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcw import ForwardCollisionWarning
from test_ipm_distance import measured


def box(height, track_id=1, x=320, bottom=390, confirmed=True):
    return {"track_id": track_id, "class_name": "car", "confidence": 0.9,
            "x1": x - 30, "x2": x + 30, "y1": bottom - height, "y2": bottom,
            "confirmed": confirmed, "observed": True}


def ground_box(distance, height=70):
    """Project a ground mark at the chosen range into a 640x480 frame."""
    pitch = math.radians(1.4)
    ny = (1.5 * math.cos(pitch) - distance * math.sin(pitch)) / (
        distance * math.cos(pitch) + 1.5 * math.sin(pitch))
    return box(height, bottom=240 + (800 * 480 / 720) * ny)


class FCWTests(unittest.TestCase):
    def test_stationary_large_vehicle_cannot_trigger_critical(self):
        warning = ForwardCollisionWarning(640, 480)
        for index in range(40):
            warning.update([box(230)], index * 0.1)
        self.assertEqual(warning.level, "CAUTION")
        self.assertAlmostEqual(warning.primary["dist_m"], 4.873, places=2)
        self.assertIsNone(warning.primary["tau"])

    def test_box_growth_without_ground_motion_cannot_raise_ttc(self):
        warning = ForwardCollisionWarning(640, 480)
        for index in range(14):
            warning.update([box(60 + index * 12, bottom=315)], index * 0.1)
        self.assertIsNone(warning.primary["tau"])
        self.assertNotIn(warning.level, ("WARNING", "DANGER"))

    def test_approach_and_ttc_threshold(self):
        warning = ForwardCollisionWarning(640, 480)
        warning.danger_ttc = 2.2
        levels = []
        for index in range(20):
            t = index * 0.1
            warning.update([ground_box(28 - 10 * t)], t)
            levels.append(warning.level)
        self.assertIn("DANGER", levels)
        self.assertIsNotNone(warning.primary["tau"])
        self.assertGreater(warning.primary["closing_mps"], 1)

    def test_missing_or_unconfirmed_observations_do_not_become_ttc(self):
        warning = ForwardCollisionWarning(640, 480)
        for index in range(10):
            warning.update([box(120 + index * 10, confirmed=False)], index * 0.1)
        self.assertIsNone(warning.primary)
        warning.update([box(100)], 1.0)
        warning.update([], 1.1)
        warning.update([box(160)], 2.1)  # gap invalidates growth history
        self.assertIsNone(warning.primary["tau"])

    def test_path_is_the_measured_ground_corridor(self):
        warning = ForwardCollisionWarning(640, 480, ipm_config=measured())
        self.assertTrue(warning.ipm.valid)
        for index in range(10):
            warning.update([box(100, x=470)], index * 0.1)
        self.assertIsNone(warning.primary)
        warning.update([box(100, x=320)], 1.0)
        self.assertIsNotNone(warning.primary["dist_m"])


if __name__ == "__main__":
    unittest.main()
