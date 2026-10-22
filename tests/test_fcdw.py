"""FCDW requires fresh ego-motion evidence and observed stable lead identity."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcdw import FrontCarDepartureWarning
from test_fcw import box


class FCDWTests(unittest.TestCase):
    def test_unknown_or_moving_speed_disables_alert(self):
        warning = FrontCarDepartureWarning(640, 480)
        for index in range(40):
            warning.update([box(100)], index * 0.1)
        self.assertEqual(warning.state, "IDLE")
        for index in range(30):
            warning.update([box(100)], index * 0.1, ego_stopped=True)
        self.assertEqual(warning.state, "TRACKING_STOPPED")
        warning.update([box(80)], 3.1, ego_stopped=False)
        self.assertEqual(warning.state, "IDLE")
        self.assertFalse(warning.alert_active)

    def test_stopped_lead_departure_and_loss(self):
        warning = FrontCarDepartureWarning(640, 480)
        for index in range(35):
            warning.update([box(100, bottom=350)], index * 0.1, ego_stopped=True)
        self.assertEqual(warning.state, "TRACKING_STOPPED")
        for index in range(10):
            warning.update([box(72, bottom=305)], 3.5 + index * 0.1, ego_stopped=True)
        self.assertTrue(warning.alert_active)
        warning.update([], 4.6, ego_stopped=None)
        self.assertFalse(warning.alert_active)

    def test_unconfirmed_lead_does_not_arm(self):
        warning = FrontCarDepartureWarning(640, 480)
        for index in range(20):
            warning.update([box(100, confirmed=False)], index * 0.1, ego_stopped=True)
        self.assertEqual(warning.state, "IDLE")


if __name__ == "__main__":
    unittest.main()
