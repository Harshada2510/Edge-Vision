"""Ground-plane IPM: provisional camera ray intersection or measured homography.

Ground coordinates are [right_m, forward_m] relative to a declared origin.
An estimated camera model is usable for demos but never marked measured.
"""
import json
import math
from pathlib import Path
import cv2
import numpy as np


class IPMDistanceEstimator:
    def __init__(self, frame_width, frame_height, calibration=None):
        self.frame_w, self.frame_h = frame_width, frame_height
        self.valid = False
        self.quality = "unavailable"
        self.reason = "No ground-plane model"
        self.H = None
        self.camera_matrix = None
        self.distortion = None
        if calibration is None:
            calibration = json.loads((Path(__file__).resolve().parent / "config" /
                                      "estimated_camera.json").read_text())
        if calibration.get("mode") == "disabled":
            self.reason = "IPM explicitly disabled"
            return
        if calibration.get("mode") == "pinhole_ground":
            self._init_estimated(calibration)
            return
        if calibration.get("measured") is not True:
            raise ValueError("Calibration must declare measured=true after physical measurement")
        if not isinstance(calibration.get("origin"), str) or not calibration["origin"].strip():
            raise ValueError("Calibration needs a named vehicle origin for metric distances")
        size = np.asarray(calibration["image_size"], dtype=float)
        image = np.asarray(calibration["image_points"], dtype=float)
        ground = np.asarray(calibration["ground_points_m"], dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid calibration image_size")
        if image.shape != (4, 2) or ground.shape != (4, 2):
            raise ValueError("Four corresponding image and ground points are required")
        if not np.isfinite(image).all() or not np.isfinite(ground).all():
            raise ValueError("Calibration contains non-finite coordinates")
        if np.any(image < 0) or np.any(image >= size) or np.any(ground[:, 1] < 0):
            raise ValueError("Calibration points must be inside the image and ahead of the origin")
        scales = np.array([frame_width, frame_height]) / size
        if abs(scales[0] / scales[1] - 1) > 0.005:
            raise ValueError("Calibration aspect ratio differs from the processed frame; recalibrate after cropping")
        image *= scales
        if "camera_matrix" in calibration:
            self.camera_matrix = np.asarray(calibration["camera_matrix"], float)
            self.distortion = np.asarray(calibration.get("dist_coeffs", []), float)
            if (self.camera_matrix.shape != (3, 3) or self.distortion.size not in (4, 5, 8, 12, 14)
                    or not np.isfinite(self.camera_matrix).all() or not np.isfinite(self.distortion).all()
                    or self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0):
                raise ValueError("Invalid lens calibration")
            self.camera_matrix = np.diag([scales[0], scales[1], 1]) @ self.camera_matrix
            image = self._undistort(image)
        elif calibration.get("lens_corrected") is not True:
            raise ValueError("Supply camera_matrix/dist_coeffs, or explicitly declare lens_corrected=true")
        for points in (image, ground):
            polygon = points.astype(np.float32)
            if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1e-3:
                raise ValueError("Calibration points must form a non-degenerate, ordered convex quadrilateral")
        self.polygon = image.astype(np.float32)
        self.H = cv2.getPerspectiveTransform(image.astype(np.float32), ground.astype(np.float32))
        if not np.isfinite(self.H).all() or abs(np.linalg.det(self.H)) < 1e-12:
            raise ValueError("Singular calibration homography")
        self.H_inv = np.linalg.inv(self.H)
        self.origin = calibration["origin"]
        self.quality = "measured"
        self.valid, self.reason = True, "Measured planar calibration loaded"

    def _init_estimated(self, config):
        """Apply the previously agreed video-derived camera guess at any image size.

        Different aspect ratios are independently rescaled; they remain estimates,
        and a new physical camera requires its own measurements for reliable range.
        """
        size = np.asarray(config["image_size"], float)
        k = np.asarray(config["camera_matrix"], float)
        height, pitch = float(config["height_m"]), math.radians(float(config["pitch_deg"]))
        max_range = float(config.get("max_distance_m", 60))
        bottom = float(config.get("max_contact_y_fraction", 0.92))
        if (config.get("quality") != "estimated" or size.shape != (2,)
                or not np.isfinite(size).all() or np.any(size <= 0)
                or k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0
                or k[1, 1] <= 0 or not np.isfinite([height, pitch, max_range, bottom]).all()
                or not 0.5 <= height <= 4 or not -0.2 <= pitch <= 0.4
                or not 10 <= max_range <= 150 or not 0.7 < bottom <= 1):
            raise ValueError("Invalid provisional pinhole camera parameters")
        if not isinstance(config.get("origin"), str) or not config["origin"].strip():
            raise ValueError("Provisional camera requires a defined ground origin")
        sx, sy = self.frame_w / size[0], self.frame_h / size[1]
        self.fx, self.fy = k[0, 0] * sx, k[1, 1] * sy
        self.cx, self.cy = k[0, 2] * sx, k[1, 2] * sy
        self.height_m, self.sin_pitch, self.cos_pitch = height, math.sin(pitch), math.cos(pitch)
        self.max_distance_m, self.max_contact_y_fraction = max_range, bottom
        self.origin = config["origin"]
        self.quality = "estimated"
        self.valid = True
        aspect_changed = abs((self.frame_w / self.frame_h) / (size[0] / size[1]) - 1) > 0.01
        self.reason = (f"ESTIMATED from video: assumed {height:g} m height, "
                       f"{k[0, 0]:g} px horizontal focal length, {math.degrees(pitch):g}° pitch"
                       + ("; image aspect differs from reference" if aspect_changed else ""))

    def _undistort(self, points):
        if self.camera_matrix is None:
            return np.asarray(points, float).reshape(-1, 2)
        return cv2.undistortPoints(np.asarray(points, float).reshape(-1, 1, 2),
                                  self.camera_matrix, self.distortion,
                                  P=self.camera_matrix).reshape(-1, 2)

    def project_to_ground(self, x, y):
        """Return (right_m, forward_m), or (None,None) outside the measured patch."""
        if not self.valid or not np.isfinite([x, y]).all():
            return None, None
        if self.quality == "estimated":
            if not 0 <= x < self.frame_w or not 0 <= y < self.frame_h * self.max_contact_y_fraction:
                return None, None
            nx, ny = (x - self.cx) / self.fx, (y - self.cy) / self.fy
            denominator = self.sin_pitch + self.cos_pitch * ny
            if denominator <= 1e-5:
                return None, None
            right = self.height_m * nx / denominator
            forward = self.height_m * (self.cos_pitch - self.sin_pitch * ny) / denominator
            if not np.isfinite([right, forward]).all() or not 1 <= forward <= self.max_distance_m:
                return None, None
            return float(right), float(forward)
        point = self._undistort([[x, y]])[0]
        if cv2.pointPolygonTest(self.polygon, tuple(map(float, point)), False) < 0:
            return None, None
        p = self.H @ np.array([*point, 1.0])
        if abs(p[2]) < 1e-9:
            return None, None
        right, forward = p[:2] / p[2]
        if not np.isfinite([right, forward]).all() or forward < 0:
            return None, None
        return float(right), float(forward)

    def estimate_distance(self, x1, y1, x2, y2):
        if x2 <= x1 or y2 <= y1 or y2 >= self.frame_h - 1:
            return None, None
        right, forward = self.project_to_ground((x1 + x2) / 2, y2)
        return forward, right

    def in_path(self, det):
        x = (det["x1"] + det["x2"]) / 2
        y = det["y2"]
        if self.valid:
            right, forward = self.project_to_ground(x, y)
            return right is not None and abs(right) <= 1.2 and forward > 0
        return False
