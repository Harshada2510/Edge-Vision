"""Import-safe per-frame ADAS pipeline and cockpit telemetry.

Detector and tracker output must be fresh observations of the same captured
frame. The runner passes video presentation time to risk engines and wall time
only to the UI. No cached/predicted detections are treated as observations.
"""
from collections import deque
import time

from fcw import ForwardCollisionWarning
from fcdw import FrontCarDepartureWarning
from ldw import LaneDepartureWarning
from alert_manager import AlertManager
from hud_renderer import HUDRenderer
from web_server import default_telemetry


class ADASPipeline:
    def __init__(self, frame_width, frame_height, enable_lane_detection=False,
                 ipm_config=None, lane_period_s=0.35, bonnet_bottom_frac=0.90,
                 hud_style="compact", lane_overlay=False, **_legacy):
        if hud_style not in ("compact", "full", "off"):
            raise ValueError("HUD style must be compact, full or off")
        self.frame_w, self.frame_h = frame_width, frame_height
        self.hud_style = hud_style
        self.fcw = ForwardCollisionWarning(frame_width, frame_height, ipm_config=ipm_config,
                                           bonnet_bottom_frac=bonnet_bottom_frac)
        self.fcdw = FrontCarDepartureWarning(frame_width, frame_height,
                                             path_filter=self.fcw.ipm.in_path)
        self.ldw = LaneDepartureWarning(frame_width, frame_height)
        self.alerts = AlertManager(alert_cooldown=2.0)
        self.hud = HUDRenderer(frame_width, frame_height)
        self.hud_layers = {"bboxes": True, "lanes": True,
                           "radar": hud_style == "full", "telemetry": hud_style == "full"}
        self.settings = dict(fcw_enabled=True, ldw_enabled=True, fcdw_enabled=True,
                             ttc_danger_threshold=1.8)
        self.enable_lane_detection = enable_lane_detection
        self.lane_overlay = lane_overlay
        self.lane_debug_mask = None
        self.lane_period_s = lane_period_s
        self.lane_source = None  # callable(now); alternative lane mask adapter
        self.last_lane_time = None
        self.frame_count = 0
        self._completion_times = deque(maxlen=31)
        self.inference_ms = self.pipeline_ms = self.hud_render_ms = 0.0

    def reset(self):
        """Call on EOF loops, seeks, camera reconnection or clock discontinuity."""
        self.fcw.reset()
        self.fcdw.reset(clear_cooldown=True)
        self.ldw.reset()
        self.alerts = AlertManager(alert_cooldown=2.0)
        self.last_lane_time = None
        self.lane_debug_mask = None
        self._completion_times.clear()

    def apply_settings(self, settings):
        for key in ("fcw_enabled", "ldw_enabled", "fcdw_enabled"):
            if key in settings:
                if type(settings[key]) is not bool:
                    raise ValueError(f"{key} must be a boolean")
                if not settings[key] and self.settings[key]:
                    {"fcw_enabled": self.fcw.reset, "fcdw_enabled": self.fcdw.reset,
                     "ldw_enabled": self.ldw.reset}[key]()
                self.settings[key] = settings[key]
        if "ttc_danger_threshold" in settings:
            value = float(settings["ttc_danger_threshold"])
            if not 0.5 <= value <= 3.0:
                raise ValueError("TTC danger threshold must be within 0.5–3.0 seconds")
            self.settings["ttc_danger_threshold"] = value
            self.fcw.danger_ttc = value

    def set_hud_layers(self, layers):
        for key in self.hud_layers:
            if key in layers:
                if type(layers[key]) is not bool:
                    raise ValueError(f"hud_{key} must be a boolean")
                self.hud_layers[key] = layers[key]

    @property
    def current_fps(self):
        if len(self._completion_times) < 2:
            return None
        span = self._completion_times[-1] - self._completion_times[0]
        return (len(self._completion_times) - 1) / span if span > 0 else None

    def process_frame(self, frame, detections, inference_ms, now, camera_source="CAM 0",
                      ego_stopped=None):
        """Risk clocks use source PTS for files; ego_stopped is None unless measured."""
        started = time.perf_counter()
        self.frame_count += 1
        self.inference_ms = float(inference_ms)
        if self.settings["fcw_enabled"]:
            self.fcw.update(detections, now)
        else:
            self.fcw.reset()
        if self.settings["fcdw_enabled"]:
            self.fcdw.update(detections, now, ego_stopped=ego_stopped)
        else:
            self.fcdw.reset()

        if self.settings["ldw_enabled"] and self.enable_lane_detection:
            if self.last_lane_time is None or now - self.last_lane_time >= self.lane_period_s:
                lane_input = self.lane_source(now) if self.lane_source else frame
                self.ldw.update(lane_input, now)
                self.lane_debug_mask = (lane_input.get("mask") if self.lane_overlay
                                        and isinstance(lane_input, dict) else None)
                self.last_lane_time = now
            else:
                self.ldw.expire(now)
        else:
            self.ldw.reset()
            self.last_lane_time = None
            self.lane_debug_mask = None
        direction = ("DRIFTING " + self.ldw.departure_direction
                     if self.ldw.alert_active else "NORMAL")
        hud_lane_state = (direction if self.ldw.lanes_detected else "NO LANES")
        active_alert = self.alerts.update(
            fcw_level=self.fcw.level, fcdw_active=self.fcdw.alert_active,
            fcdw_message=self.fcdw.alert_message, ldw_active=self.ldw.alert_active,
            ldw_message=self.ldw.alert_message, ldw_dir=self.ldw.departure_direction, now=now)

        primary = self.fcw.primary
        lead_id = primary["track_id"] if primary else None
        objects = []
        for det in detections:
            box = tuple(det[k] for k in ("x1", "y1", "x2", "y2"))
            in_path = self.fcw.ipm.in_path(det)
            dist, lat = self.fcw.ipm.estimate_distance(*box)
            selected = lead_id is not None and lead_id == det["track_id"]
            objects.append({"track_id": det["track_id"], "name": det["class_name"],
                            "confidence": det["confidence"], "box": box,
                            "confirmed": det.get("confirmed", True),
                            "risk": self.fcw.level if selected else "SAFE",
                            "distance_m": dist, "lateral_offset_m": lat,
                            "ttc": primary.get("tau") if selected else None,
                            "in_ahead": in_path, "in_path": in_path})
        lane_lines = ((tuple(self.ldw.left_line), tuple(self.ldw.right_line))
                      if self.ldw.lanes_detected else None)
        hud_started = time.perf_counter()
        self.hud.render(frame, tracked_objects=objects, lane_lines=lane_lines,
                        ldw_direction=hud_lane_state, active_alert=active_alert,
                        fps=self.current_fps, inference_ms=self.inference_ms,
                        lead_vehicle_id=lead_id,
                        show_bboxes=self.hud_layers["bboxes"],
                        show_lanes=self.hud_layers["lanes"],
                        show_radar=self.hud_layers["radar"] and self.fcw.ipm.valid,
                        show_telemetry=self.hud_layers["telemetry"],
                        camera_source=camera_source, now=now,
                        status_text=("IPM ~ ESTIMATED" if self.fcw.ipm.quality == "estimated" else
                                     None if self.fcw.ipm.valid else "IPM UNAVAILABLE"),
                        metric_estimated=self.fcw.ipm.quality == "estimated",
                        ldw_available=self.settings["ldw_enabled"] and self.enable_lane_detection,
                        fcdw_available=self.settings["fcdw_enabled"] and ego_stopped is not None,
                        style=self.hud_style,
                        lane_debug_mask=(self.lane_debug_mask if self.lane_overlay
                                         and self.last_lane_time is not None
                                         and 0 <= now - self.last_lane_time <= 0.6 else None),
                        lane_debug_age=(now - self.last_lane_time if self.last_lane_time is not None
                                        else None))
        self.hud_render_ms = (time.perf_counter() - hud_started) * 1000
        self.pipeline_ms = (time.perf_counter() - started) * 1000
        self._completion_times.append(time.perf_counter())
        stopped_duration = (max(0, now - self.fcdw.stopped_start_time)
                            if self.fcdw.state == "TRACKING_STOPPED"
                            and self.fcdw.stopped_start_time is not None else 0.0)
        tel = default_telemetry()
        tel.update({
            "timestamp": time.time(), "source_time_s": now,
            "fps": self.current_fps, "inference_ms": self.inference_ms,
            "pipeline_ms": self.pipeline_ms, "hud_render_ms": self.hud_render_ms,
            "frame_size": [frame.shape[1], frame.shape[0]], "track_count": len(objects),
            "camera_source": camera_source, "active_alert": active_alert,
            "calibration": {"valid": self.fcw.ipm.valid, "quality": self.fcw.ipm.quality,
                            "reason": self.fcw.ipm.reason},
            "fcw": {"level": self.fcw.level, "distance_m": primary.get("dist_m") if primary else None,
                    "lateral_offset_m": primary.get("lat_m") if primary else None,
                    "ttc": primary.get("tau") if primary else None,
                    "lead_class": primary["class_name"] if primary else None,
                    "lead_track_id": lead_id, "enabled": self.settings["fcw_enabled"],
                    "closing_mps": primary.get("closing_mps") if primary else None,
                    "risk_basis": "ipm_ground_distance_and_closing",
                    "path_basis": (self.fcw.ipm.quality + "_ground"
                                   if self.fcw.ipm.valid else "unavailable")},
            "ldw": {"departure_direction": direction, "offset": round(self.ldw.lateral_offset, 3),
                    "status": self.ldw.status, "lanes_detected": self.ldw.lanes_detected,
                    "alert_message": self.ldw.alert_message,
                    "enabled": self.settings["ldw_enabled"] and self.enable_lane_detection},
            "fcdw": {"state": self.fcdw.state, "alert_active": self.fcdw.alert_active,
                     "alert_message": self.fcdw.alert_message, "stopped_duration": stopped_duration,
                     "lead_track_id": self.fcdw.lead_info["track_id"] if self.fcdw.lead_info else None,
                     "enabled": self.settings["fcdw_enabled"],
                     "ego_stopped": ego_stopped,
                     "availability": "speed_valid" if ego_stopped is not None else "speed_unavailable"},
            "objects": objects, "hud_layers": dict(self.hud_layers),
        })
        return tel
