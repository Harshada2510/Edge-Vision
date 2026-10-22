"""Forward warning based on IPM road range and closing speed from fresh tracks.

Estimated camera geometry provides provisional demonstration distances only;
its TTC/alerts are not validated collision predictions.
"""

from ipm_distance import IPMDistanceEstimator

# Classes that matter for a forward collision risk (COCO names).
RELEVANT_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle", "person",
}

# Bottom box centre approximates ground contact; IPM defines the corridor.
HORIZON_FRACTION = 0.35          # top 35% of the frame = sky / far away

# Ignore boxes smaller than this fraction of frame height (noise floor).
MIN_HEIGHT_FRACTION = 0.04

RANGE_MIN_SPAN = 0.50        # seconds between ground-range samples
TTC_MAX = 20.0

# Ground-range TTC thresholds (seconds); exits are looser for hysteresis.
CAUTION_TTC_ENTER = 10.0
CAUTION_TTC_EXIT = 12.0
WARNING_TTC_ENTER = 3.5
WARNING_TTC_EXIT = 7.0
DANGER_TTC_ENTER = 1.8
DANGER_TTC_EXIT = 3.0

# Debounce: consecutive frames a candidate level must hold before applying.
# Hardening into a worse level is faster than releasing.
CONFIRM_WORSE_FRAMES = 3
CONFIRM_BETTER_FRAMES = 8

# How long (s) a vanished track's state is kept before it is forgotten.
TRACK_FORGET_SECONDS = 2.0

LEVEL_ORDER = ["DANGER", "WARNING", "CAUTION", "SAFE"]
LEVEL_INDEX = {level: i for i, level in enumerate(LEVEL_ORDER)}


class ForwardCollisionWarning:
    """Tracks relevant objects and produces one stable FCW level per frame."""

    def __init__(self, frame_width, frame_height, cam_height_m=1.25,
                 pitch_angle_deg=5.0, ipm_config=None, bonnet_bottom_frac=0.90):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.bonnet_bottom_frac = bonnet_bottom_frac

        # Explicit measured calibration overrides the provisional default.
        # Legacy height/pitch kwargs are ignored so only one camera model is
        # active and there are no inert live settings.
        self.ipm = IPMDistanceEstimator(frame_width, frame_height, calibration=ipm_config)
        self.danger_ttc = DANGER_TTC_ENTER

        # Per-track timestamped ground range. Predicted or unconfirmed boxes
        # never reach this module through its observation gate.
        self._track_state = {}

        # Global (per "driver") state machine.
        self.level = "SAFE"
        self._candidate = "SAFE"
        self._confirm_count = 0

        # Info about the most dangerous object (for display).
        self.primary = None
        self.last_update = None

    def reset(self):
        self._track_state.clear()
        self.level, self._candidate, self._confirm_count = "SAFE", "SAFE", 0
        self.primary = None
        self.last_update = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections, now):
        """
        detections: list of dicts with keys
            track_id, class_name, confidence, x1, y1, x2, y2 (ints, pixels)
        now: float — monotonic time of this frame.

        Returns self with:
            .level   -> "SAFE" | "CAUTION" | "WARNING" | "DANGER"
            .primary -> dict for the closest relevant object or None:
                        {track_id, class_name, confidence, box, dist_m, tau}
        """
        if self.last_update is not None and (now <= self.last_update or now - self.last_update > 1.0):
            self.reset()
        self.last_update = now
        relevant = [d for d in detections if d.get("confirmed", True) and d.get("observed", True)
                    and self._is_relevant(d)]

        self._prune_tracks(relevant, now)

        risks = [
            r for r in (self._evaluate_object(d, now) for d in relevant)
            if r is not None
        ]

        # The worst in-path object drives FCW; at equal risk use shorter range.
        if risks:
            self.primary = min(
                risks,
                key=lambda r: (r["natural_rank"], r["dist_m"]),
            )
        else:
            self.primary = None

        self._advance_state()
        return self

    @property
    def is_active(self):
        return self.level in ("CAUTION", "WARNING", "DANGER")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_relevant(self, det):
        """Class + geometry filter: is this object in our driving path?"""
        if det.get("class_name", "") not in RELEVANT_CLASSES:
            return False

        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        width = x2 - x1
        height = y2 - y1

        if width < 1 or height < 1:
            return False
        if y2 >= self.frame_h * self.bonnet_bottom_frac and width >= 0.35 * self.frame_w:
            return False  # no observable ground contact; likely ego bonnet
        if height / self.frame_h < MIN_HEIGHT_FRACTION:
            return False

        # Road contact point = bottom edge centre of the box.
        bottom_y = y2

        if bottom_y < self.frame_h * HORIZON_FRACTION:
            return False  # above the horizon -> not on our road

        return self.ipm.in_path(det)

    def _evaluate_object(self, det, now):
        """Ground range and TTC for one confirmed observation."""
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        height = y2 - y1
        if height <= 0:
            return None
        track_id = det["track_id"]
        h_frac = height / self.frame_h

        dist_m, lat_m = self.ipm.estimate_distance(x1, y1, x2, y2)
        if dist_m is None:
            return None
        st = self._track_state.setdefault(track_id, {"samples": [], "closing_smooth": 0.0})
        if st["samples"] and now - st["samples"][-1][0] > 0.6:
            st["samples"].clear()
            st["closing_smooth"] = 0.0
        st["samples"].append((now, dist_m))
        if len(st["samples"]) > 60:
            st["samples"].pop(0)

        # Ground-plane distance over at least 0.5 s of the same confirmed
        # track. Implausibly large changes indicate association/box jitter.
        tau = None
        closing = 0.0
        samples = st["samples"]
        if len(samples) >= 2:
            old = None
            for s in reversed(samples):
                if now - s[0] >= RANGE_MIN_SPAN:
                    old = s
                    break
            if old is not None:
                dt = now - old[0]
                if dt > 0:
                    raw_closing = (old[1] - dist_m) / dt
                    if abs(raw_closing) <= 25.0:
                        st["closing_smooth"] = (0.4 * raw_closing +
                                                0.6 * st["closing_smooth"])
                        closing = st["closing_smooth"]
                        if closing > 1.0:
                            tau = dist_m / closing
                            if tau > TTC_MAX:
                                tau = None
                    else:
                        st["closing_smooth"] = 0.0

        return {
            "track_id": track_id,
            "class_name": det["class_name"],
            "confidence": det.get("confidence", 0.0),
            "box": (x1, y1, x2, y2),
            "h_frac": h_frac,
            "tau": tau,
            "closing_mps": closing,
            "dist_m": dist_m,
            "lat_m": lat_m,
            "natural_rank": self._enter_rank(dist_m, tau, closing),
        }

    def _enter_rank(self, dist_m, tau, closing):
        """Only ground-distance/TTC evidence can raise a forward warning."""
        if tau is not None and tau < self.danger_ttc and dist_m < 20 and closing > 1.5:
            return LEVEL_INDEX["DANGER"]
        if tau is not None and tau < WARNING_TTC_ENTER and dist_m < 35:
            return LEVEL_INDEX["WARNING"]
        if dist_m < 22 or (tau is not None and tau < CAUTION_TTC_ENTER and dist_m < 45):
            return LEVEL_INDEX["CAUTION"]
        return LEVEL_INDEX["SAFE"]

    def _hysteresis_rank(self, current_rank, natural_rank, dist_m, tau):
        """
        Apply the EXIT thresholds when the situation is improving.
        A non-active level with a worsening object applies immediately
        (chip away) — the confirmation counter below does the debouncing.
        When improving, the level may drop only ONE step, and only if the
        current level's EXIT conditions are met (looser than ENTER, so the
        warning does not clear on a borderline value).
        """
        if natural_rank < current_rank:
            return natural_rank  # worsening: use ENTER conditions directly

        if current_rank == LEVEL_INDEX["DANGER"]:
            if dist_m is None or dist_m > 23 or tau is None or tau > max(DANGER_TTC_EXIT, self.danger_ttc + 0.7):
                return LEVEL_INDEX["WARNING"]
            return LEVEL_INDEX["DANGER"]
        if current_rank == LEVEL_INDEX["WARNING"]:
            if dist_m is None or dist_m > 39 or tau is None or tau > WARNING_TTC_EXIT:
                return LEVEL_INDEX["CAUTION"]
            return LEVEL_INDEX["WARNING"]
        if current_rank == LEVEL_INDEX["CAUTION"]:
            if dist_m is None or (dist_m > 27 and (tau is None or tau > CAUTION_TTC_EXIT)):
                return LEVEL_INDEX["SAFE"]
            return LEVEL_INDEX["CAUTION"]
        return LEVEL_INDEX["SAFE"]

    def _advance_state(self):
        """Debounced state machine.

        Worsening candidates need CONFIRM_WORSE_FRAMES consecutive frames,
        improving candidates need CONFIRM_BETTER_FRAMES (slow release).
        """
        if self.primary is None:
            natural_rank = LEVEL_INDEX["SAFE"]
            dist_m = None
            tau = None
        else:
            natural_rank = self.primary["natural_rank"]
            dist_m = self.primary["dist_m"]
            tau = self.primary["tau"]

        current_rank = LEVEL_INDEX[self.level]
        candidate_rank = self._hysteresis_rank(
            current_rank, natural_rank, dist_m, tau
        )
        candidate = LEVEL_ORDER[candidate_rank]

        if candidate_rank == current_rank:
            self._candidate = candidate
            self._confirm_count = 0
            return

        worsening = candidate_rank < current_rank  # lower index = worse
        needed = (
            CONFIRM_WORSE_FRAMES
            if worsening
            else CONFIRM_BETTER_FRAMES
        )

        if candidate != self._candidate:
            self._candidate = candidate
            self._confirm_count = 1
        else:
            self._confirm_count += 1

        if self._confirm_count >= needed:
            self.level = candidate
            self._candidate = candidate
            self._confirm_count = 0

    def _prune_tracks(self, relevant, now):
        """Forget track state for objects not seen for TRACK_FORGET_SECONDS."""
        seen = {d["track_id"] for d in relevant}
        cutoff = now - TRACK_FORGET_SECONDS
        for tid in list(self._track_state):
            st = self._track_state[tid]
            last_seen = st["samples"][-1][0] if st["samples"] else 0.0
            if tid not in seen and last_seen < cutoff:
                del self._track_state[tid]
