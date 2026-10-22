"""Optional TriLiteNet Tiny lane segmentation head on NCNN CPU.

This independently converted graph is experimental. It extracts only the lane
head (out4), not the detector or drivable-area heads. A two-sided consistent
mask is required for a lane warning; unavailable masks produce no warning.
"""
import cv2
import numpy as np

from detector import NCNNModel


class TriLiteLane(NCNNModel):
    def __init__(self, model_dir, threads=3):
        super().__init__(model_dir, "trilitenet-tiny-320", threads)

    def __call__(self, frame, include_mask=False):
        h, w = frame.shape[:2]
        net_w, net_h = 320, 192
        scale = min(net_w / w, net_h / h)
        rw, rh = round(w * scale), round(h * scale)
        px, py = (net_w - rw) // 2, (net_h - rh) // 2
        mat = self.ncnn.Mat.from_pixels_resize(
            np.ascontiguousarray(frame), self.ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, rw, rh)
        mat = self.ncnn.copy_make_border(mat, py, net_h - rh - py,
                                         px, net_w - rw - px,
                                         self.ncnn.BorderType.BORDER_CONSTANT, 114)
        mat.substract_mean_normalize([123.675, 116.28, 103.53],
                                     [1 / 58.395, 1 / 57.12, 1 / 57.375])
        logits = self.extract(mat, "out4")
        if logits.shape != (2, net_h, net_w):
            raise ValueError(f"Unexpected lane head shape {logits.shape}")
        mask = (logits[1] > logits[0]).astype(np.uint8)[py:py + rh, px:px + rw]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        result = lane_lines(mask)
        if include_mask:
            # Reuse the resized binary mask already needed for fitting.
            # Only visual inspection retains it between lane updates.
            result["mask"] = mask
        return result


def lane_lines(mask, top_fraction=0.52, bottom_fraction=0.76):
    """Fit coherent components crossing several image rows; reject fragments."""
    h, w = mask.shape
    top, bottom = round(h * top_fraction), round(h * bottom_fraction)
    region = np.ascontiguousarray((mask[top:bottom] > 0).astype(np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    sides = {"left": None, "right": None}
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < 25 or ch < (bottom - top) * 0.6 or cw > w * 0.35:
            continue
        ys, xs = np.where(labels == i)
        # Components must have marks near both sampling rows.
        upper = xs[ys < (bottom - top) * 0.35]
        lower = xs[ys > (bottom - top) * 0.65]
        if len(upper) < 5 or len(lower) < 5:
            continue
        low, high = int(np.median(lower)), int(np.median(upper))
        side = "left" if low < w / 2 else "right"
        line = (low, bottom, high, top)
        old = sides[side]
        if old is None or abs(low - w / 2) < abs(old[0] - w / 2):
            sides[side] = line
    left, right = sides["left"], sides["right"]
    if left is None or right is None or not 0.18 * w < right[0] - left[0] < 0.9 * w:
        return {"left_line": None, "right_line": None}
    return {"left_line": left, "right_line": right}
