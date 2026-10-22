"""Ground-plane projection tests using known physical correspondences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from ipm_distance import IPMDistanceEstimator


def measured():
    # Synthetic perspective: far edge y=200 is z=20m; near y=400 is z=2m.
    return {"measured": True, "origin": "camera contact point on road",
            "lens_corrected": True, "image_size": [640, 480],
            "image_points": [[260, 200], [380, 200], [500, 400], [140, 400]],
            "ground_points_m": [[-2, 20], [2, 20], [2, 2], [-2, 2]]}


class IPMTests(unittest.TestCase):
    def test_default_estimated_ground_and_explicit_disable(self):
        ipm = IPMDistanceEstimator(640, 480)
        self.assertTrue(ipm.valid)
        self.assertEqual(ipm.quality, "estimated")
        near, lateral = ipm.estimate_distance(270, 220, 370, 390)
        self.assertAlmostEqual(near, 4.873, places=2)
        self.assertAlmostEqual(lateral, 0, places=3)
        self.assertEqual(ipm.project_to_ground(320, 220), (None, None))
        disabled = IPMDistanceEstimator(640, 480, {"mode": "disabled"})
        self.assertFalse(disabled.valid)
        self.assertEqual(disabled.estimate_distance(270, 220, 370, 390), (None, None))
        with self.assertRaises(ValueError):
            IPMDistanceEstimator(640, 480, {**measured(), "measured": False})

    def test_known_ground_correspondences(self):
        ipm = IPMDistanceEstimator(640, 480, measured())
        for pixel, ground in zip(measured()["image_points"], measured()["ground_points_m"]):
            result = ipm.project_to_ground(*pixel)
            for observed, expected in zip(result, ground):
                self.assertAlmostEqual(observed, expected, places=4)
        near = ipm.estimate_distance(270, 250, 370, 390)
        far = ipm.estimate_distance(290, 205, 350, 230)
        self.assertLess(near[0], far[0])
        self.assertAlmostEqual(near[1], 0, places=4)

    def test_outside_plane_rejected(self):
        ipm = IPMDistanceEstimator(640, 480, measured())
        self.assertEqual(ipm.project_to_ground(320, 50), (None, None))
        self.assertEqual(ipm.project_to_ground(20, 350), (None, None))
        self.assertEqual(ipm.estimate_distance(100, 200, 200, 479), (None, None))

    def test_scaled_frame_and_aspect_change(self):
        ipm = IPMDistanceEstimator(320, 240, measured())
        self.assertAlmostEqual(ipm.project_to_ground(130, 100)[1], 20, places=4)
        with self.assertRaises(ValueError):
            IPMDistanceEstimator(640, 360, measured())

    def test_rejects_unmeasured_geometry_and_lens(self):
        config = measured()
        config["image_points"] = [[300, 300]] * 4
        with self.assertRaises(ValueError):
            IPMDistanceEstimator(640, 480, config)
        config = measured()
        del config["lens_corrected"]
        with self.assertRaises(ValueError):
            IPMDistanceEstimator(640, 480, config)


if __name__ == "__main__":
    unittest.main()
