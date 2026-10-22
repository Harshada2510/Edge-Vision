"""Direct NCNN CPU inference without Torch or Ultralytics at runtime.

FastestDet preprocessing/decoder follows dog-qiuqiu/FastestDet (BSD-3-Clause).
See models/manifest.json and THIRD_PARTY.md for pinned weights and attribution.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

LABELS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def nms(boxes, threshold):
    """Class-aware NMS on [x1,y1,x2,y2,score,class]; bound candidate work."""
    boxes = boxes[np.argsort(-boxes[:, 4])[:1000]]
    keep = []
    while len(boxes):
        best, rest = boxes[0], boxes[1:]
        keep.append(best)
        lo = np.maximum(best[:2], rest[:, :2])
        hi = np.minimum(best[2:4], rest[:, 2:4])
        overlap = np.prod(np.maximum(hi - lo, 0), axis=1)
        area = np.prod(np.maximum(rest[:, 2:4] - rest[:, :2], 0), axis=1)
        union = area + np.prod(best[2:4] - best[:2]) - overlap
        boxes = rest[(overlap / np.maximum(union, 1e-8) <= threshold) |
                     (rest[:, 5] != best[5])]
    return np.asarray(keep, dtype=np.float32).reshape(-1, 6)


def decode_fastestdet(output, width, height, threshold=0.65, nms_iou=0.45,
                      bonnet_bottom_frac=0.90):
    if output.ndim != 3 or output.shape[0] != 85:
        raise ValueError(f"Expected FastestDet [85,H,W], got {output.shape}")
    classes = output[5:].argmax(axis=0)
    scores = np.maximum(output[5:].max(axis=0), 0)**0.4 * np.maximum(output[0], 0)**0.6
    yy, xx = np.where((scores >= threshold) & np.isin(classes, list(LABELS)))
    centers = (np.tanh(output[1:3, yy, xx]).T + np.stack([xx, yy], axis=-1))
    centers *= np.array([width / output.shape[2], height / output.shape[1]])
    wh = (1 / (1 + np.exp(-np.clip(output[3:5, yy, xx], -80, 80)))).T
    wh *= np.array([width, height])
    boxes = np.column_stack([centers - wh / 2, centers + wh / 2,
                             scores[yy, xx], classes[yy, xx]])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
    boxes = boxes[np.isfinite(boxes).all(axis=1) &
                  (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)]
    # Large, near-bottom boxes often describe the camera vehicle's own bonnet.
    # Its tyre contact point is not in view; the cutoff is camera adjustable.
    boxes = boxes[~((boxes[:, 3] >= height * bonnet_bottom_frac) &
                    (boxes[:, 2] - boxes[:, 0] >= width * 0.35))]
    return [dict(zip(("x1", "y1", "x2", "y2", "confidence"), map(float, row[:5])),
                 class_name=LABELS[int(row[5])]) for row in nms(boxes, nms_iou)]


class NCNNModel:
    def __init__(self, model_dir, name, threads=3):
        import ncnn
        self.ncnn = ncnn
        root = Path(model_dir)
        self.spec = json.loads((root / "manifest.json").read_text())[name]
        if not 1 <= threads <= 4:
            raise ValueError("NCNN threads must be between 1 and 4 for this Pi profile")
        ncnn.set_omp_num_threads(threads)
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_bf16_storage = False
        for kind in ("param", "bin"):
            path = root / self.spec[kind]
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.spec[kind + "_sha256"]:
                raise ValueError(f"Model checksum mismatch: {path}")
            status = self.net.load_param(str(path)) if kind == "param" else self.net.load_model(str(path))
            if status != 0:
                raise RuntimeError(f"NCNN failed to load {path}: {status}")

    def extract(self, mat, output):
        extractor = self.net.create_extractor()
        if extractor.input(self.spec["input"], mat) != 0:
            raise RuntimeError("NCNN input failed")
        status, value = extractor.extract(output)
        if status != 0:
            raise RuntimeError(f"NCNN extraction failed: {output}, {status}")
        return np.array(value, copy=True)


class FastestDet(NCNNModel):
    def __init__(self, model_dir, threads=3, threshold=0.65, bonnet_bottom_frac=0.90):
        super().__init__(model_dir, "fastestdet", threads)
        if not 0 < threshold <= 1:
            raise ValueError("Detection threshold must be in (0,1]")
        self.threshold = threshold
        self.bonnet_bottom_frac = bonnet_bottom_frac

    def __call__(self, frame):
        start = time.perf_counter()
        h, w = frame.shape[:2]
        # This model was trained using BGR and square stretching, not letterboxing.
        mat = self.ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(frame), self.ncnn.Mat.PixelType.PIXEL_BGR, w, h, 352, 352)
        mat.substract_mean_normalize([0, 0, 0], [1 / 255] * 3)
        prepared = time.perf_counter()
        output = self.extract(mat, self.spec["outputs"][0])
        inferred = time.perf_counter()
        detections = decode_fastestdet(output, w, h, self.threshold, self.spec["nms"],
                                      self.bonnet_bottom_frac)
        ended = time.perf_counter()
        return detections, {"preprocess_ms": (prepared - start) * 1000,
                            "inference_ms": (inferred - prepared) * 1000,
                            "postprocess_ms": (ended - inferred) * 1000,
                            "detector_ms": (ended - start) * 1000}
