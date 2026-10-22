"""Fresh external ego speed evidence for FCDW; missing or stale is UNKNOWN."""
import json
import math
import time
from pathlib import Path


class EgoSpeedFile:
    def __init__(self, path, max_age_s=1.0):
        self.path = Path(path)
        self.max_age_s = max_age_s

    def stopped(self):
        try:
            state = json.loads(self.path.read_text())
            stamp = float(state["unix_time"])
            speed = float(state["speed_mps"])
            age = time.time() - stamp
            if not math.isfinite(speed) or not 0 <= speed <= 80 or not 0 <= age <= self.max_age_s:
                return None
            return speed < 0.3
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
