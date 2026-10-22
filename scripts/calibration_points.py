"""Select four road-plane pixel correspondences on an unmodified source frame.

python scripts/calibration_points.py --video dashcam.mp4 --frame 100 --output config/calibration.json
Record real metres separately and edit the generated JSON before enabling it.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--export-frame", type=Path, help="Save raw image for marking in an external editor")
    args = p.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Requested frame is outside the video")
    if args.export_frame:
        args.export_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.export_frame), frame)
        print(f"Raw source frame exported to {args.export_frame}")
        return
    points = []
    h, w = frame.shape[:2]
    factor = min(1, 1100 / w, 750 / h)
    preview = cv2.resize(frame, (round(w * factor), round(h * factor)))
    def click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([round(x / factor), round(y / factor)])
            cv2.circle(preview, (x, y), 5, (0, 255, 255), -1)
    cv2.namedWindow("Pick far-left, far-right, near-right, near-left")
    cv2.setMouseCallback("Pick far-left, far-right, near-right, near-left", click)
    while len(points) < 4:
        cv2.imshow("Pick far-left, far-right, near-right, near-left", preview)
        if cv2.waitKey(30) & 0xff == 27:
            sys.exit("Cancelled before collecting four points")
    cv2.destroyAllWindows()
    data = {"measured": False,
            "origin": "Fill in the physical origin used for all ground measurements",
            "image_size": [w, h], "image_points": points,
            "ground_points_m": [None, None, None, None],
            "lens_corrected": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {args.output}; measure ground points and lens geometry before setting measured=true.")


if __name__ == "__main__":
    main()
