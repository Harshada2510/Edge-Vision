"""Small IoU/velocity tracker for vehicle detections.

Predictions assist association only. Missing detections are never returned as
fresh observations, so stale boxes cannot feed TTC, distance or departure.
"""
import numpy as np


def iou(a, b):
    overlap = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0))
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - overlap
    return float(overlap / max(union, 1e-6))


class VehicleTracker:
    def __init__(self, min_hits=3, max_age=0.6, iou_threshold=0.25):
        self.min_hits, self.max_age, self.iou_threshold = min_hits, max_age, iou_threshold
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.last_time = None

    def update(self, detections, now):
        if not np.isfinite(now):
            raise ValueError("Track timestamp must be finite")
        if self.last_time is not None and now <= self.last_time:
            self.reset()
        self.last_time = now
        self.tracks = {tid: st for tid, st in self.tracks.items() if now - st["time"] <= self.max_age}
        boxes = [np.array([d[k] for k in ("x1", "y1", "x2", "y2")], float) for d in detections]
        candidates = []
        for tid, st in self.tracks.items():
            predicted = st["box"] + st["velocity"] * min(now - st["time"], 0.25)
            for index, box in enumerate(boxes):
                score = iou(predicted, box)
                if score >= self.iou_threshold:
                    candidates.append((score, tid, index))
        matched, used = {}, set()
        for _, tid, index in sorted(candidates, reverse=True):
            if tid not in used and index not in matched:
                used.add(tid)
                matched[index] = tid
        for tid, st in self.tracks.items():
            if tid not in used:
                st["streak"] = 0
        result = []
        for index, (det, box) in enumerate(zip(detections, boxes)):
            if index in matched:
                tid = matched[index]
                st = self.tracks[tid]
                dt = max(now - st["time"], 1e-6)
                velocity = 0.5 * st["velocity"] + 0.5 * (box - st["box"]) / dt
                streak = st["streak"] + 1
            else:
                tid = self.next_id
                self.next_id += 1
                velocity, streak = np.zeros(4), 1
            self.tracks[tid] = {"box": box, "velocity": velocity, "time": now, "streak": streak}
            result.append(dict(det, track_id=tid, confirmed=streak >= self.min_hits,
                               observed=True, observation_time=now))
        return result
