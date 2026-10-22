"""Measured/unknown states make it through the actual cockpit pipeline."""
import os
import sys
from pathlib import Path
import unittest

os.environ["EDGEVISION_NO_CV2"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline import ADASPipeline
from test_ipm_distance import measured
from test_fcw import box
from web_server import TelemetryHub


def blank():
    return np.zeros((480, 640, 3), np.uint8)


class CockpitTests(unittest.TestCase):
    def test_estimated_metric_and_unknown_stop_signals_are_explicit(self):
        pipeline = ADASPipeline(640, 480)
        output = pipeline.process_frame(blank(), [box(90)], 9.0, 0.0)
        self.assertTrue(output["calibration"]["valid"])
        self.assertEqual(output["calibration"]["quality"], "estimated")
        self.assertIsNotNone(output["objects"][0]["distance_m"])
        self.assertEqual(output["fcw"]["path_basis"], "estimated_ground")
        self.assertIsNone(output["fcdw"]["ego_stopped"])
        self.assertEqual(output["fcdw"]["state"], "IDLE")
        self.assertFalse(output["ldw"]["enabled"])
        self.assertIsNone(output["fps"])  # one completed frame cannot define throughput

    def test_explicitly_disabled_geometry_has_no_metric_warning(self):
        pipeline = ADASPipeline(640, 480, ipm_config={"mode": "disabled"})
        for index in range(8):
            output = pipeline.process_frame(blank(), [box(100)], 9.0, index * .1)
        self.assertEqual(output["calibration"]["quality"], "unavailable")
        self.assertIsNone(output["objects"][0]["distance_m"])
        self.assertEqual(output["fcw"]["level"], "SAFE")

    def test_measured_geometry_and_settings_are_live(self):
        pipeline = ADASPipeline(640, 480, ipm_config=measured())
        output = pipeline.process_frame(blank(), [box(80)], 9.0, 0.0)
        self.assertTrue(output["calibration"]["valid"])
        self.assertEqual(output["calibration"]["quality"], "measured")
        self.assertIsNotNone(output["objects"][0]["distance_m"])
        pipeline.apply_settings({"ttc_danger_threshold": 2.3, "fcw_enabled": False})
        self.assertEqual(pipeline.fcw.danger_ttc, 2.3)
        output = pipeline.process_frame(blank(), [box(80)], 9.0, 0.1)
        self.assertEqual(output["fcw"]["level"], "SAFE")
        self.assertIsNone(output["fcw"]["lead_track_id"])
        self.assertGreater(output["fps"], 0)
        pipeline.reset()
        self.assertIsNone(pipeline.current_fps)

    def test_lanes_expire_and_user_toggle_persists(self):
        pipeline = ADASPipeline(640, 480, enable_lane_detection=True, lane_period_s=0.8)
        pipeline.lane_source = lambda now: {"left_line": (130, 370, 240, 245),
                                             "right_line": (510, 370, 400, 245)}
        output = pipeline.process_frame(blank(), [], 9.0, 0.0)
        self.assertTrue(output["ldw"]["lanes_detected"])
        output = pipeline.process_frame(blank(), [], 9.0, 0.7)
        self.assertFalse(output["ldw"]["lanes_detected"])
        hub = TelemetryHub()
        hub.set_hud_layer("radar", False)
        hub.publish_telemetry(output)
        self.assertFalse(hub.get_telemetry()["hud_layers"]["radar"])


if __name__ == "__main__":
    unittest.main()
