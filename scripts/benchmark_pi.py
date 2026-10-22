"""Replay the same video across NCNN thread counts and native HUD styles.

python scripts/benchmark_pi.py --video clip.mp4 --threads 2 --hud-styles full compact off --web --output results/hud.json
"""
import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--threads", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--lanes", choices=["off", "trilite", "hough"], default="off")
    p.add_argument("--lane-overlay", action="store_true",
                   help="Include the optional TriLite raw-mask visual overlay")
    p.add_argument("--hud-styles", choices=["compact", "full", "off"], nargs="+", default=["compact"])
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--web", action="store_true", help="Include JPEG encoding in timing")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.frames <= args.warmup:
        p.error("--frames must exceed --warmup")
    if args.lane_overlay and args.lanes != "trilite":
        p.error("--lane-overlay requires --lanes trilite")
    trials = []
    with tempfile.TemporaryDirectory() as folder:
        for threads in args.threads:
            for style in args.hud_styles:
                report = Path(folder) / f"threads_{threads}_{style}.json"
                cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "main.py"),
                       "--video", str(args.video), "--threads", str(threads), "--lanes", args.lanes,
                       "--hud-style", style, "--max-frames", str(args.frames),
                       "--warmup", str(args.warmup), "--metrics", str(report)]
                if args.lane_overlay:
                    cmd.append("--lane-overlay")
                if args.web:
                    cmd += ["--host", "127.0.0.1", "--port", "0"]
                else:
                    cmd += ["--no-web"]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                trial = json.loads(report.read_text())
                trials.append(trial)
                stages = trial["stages"]
                jpeg_stage = stages.get("jpeg_encode_ms")
                jpeg_time = (f"{jpeg_stage['median_ms']:.1f} ms"
                             if jpeg_stage else "n/a")
                print(f"{threads} threads / {style}: {trial['observed_fps']:.2f} fps; "
                      f"inference {stages['inference_ms']['median_ms']:.1f} ms; "
                      f"HUD {stages['hud_render_ms']['median_ms']:.1f} ms; "
                      f"JPEG/encoded frame {jpeg_time}; "
                      f"p95 frame {stages['frame_total_ms']['p95_ms']:.1f} ms", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"host": platform.platform(), "trials": trials}, indent=2) + "\n")


if __name__ == "__main__":
    main()
