"""Check the actual 64-bit Trixie Pi before starting the browser demo.

Run with the project's .venv activated; --picamera checks CSI imports and
camera discovery in addition to the common video/USB dependencies.
"""
import argparse
import importlib.metadata
import platform
from pathlib import Path
import sys


def memory_mib():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, amount = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[key] = round(int(amount.split()[0]) / 1024)
    return values


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--picamera", action="store_true", help="Also check CSI camera detection")
    args = args.parse_args(argv)
    failures = []
    arch = platform.machine()
    os_release = Path("/etc/os-release").read_text()
    trixie = any(line.split("=", 1)[1].strip('"') == "trixie"
                 for line in os_release.splitlines()
                 if line.startswith("VERSION_CODENAME="))
    print(f"Machine: {arch}; Python: {platform.python_version()}; Trixie: {trixie}")
    if arch != "aarch64":
        failures.append(f"Expected your 64-bit Pi OS to report aarch64, got {arch}")
    if not trixie:
        failures.append("Expected the confirmed Debian Trixie-based OS")
    print("Memory (MiB):", memory_mib())

    for module_name, package_name in (("numpy", "numpy"), ("cv2", None),
                                      ("flask", "flask"), ("ncnn", "ncnn")):
        try:
            module = __import__(module_name)
            try:
                version = (importlib.metadata.version(package_name) if package_name
                           else module.__version__)
            except (importlib.metadata.PackageNotFoundError, AttributeError):
                version = getattr(module, "__version__", "installed")
            print(f"{module_name}: {version} ({Path(module.__file__).resolve()})")
        except (ImportError, OSError, AttributeError, importlib.metadata.PackageNotFoundError) as exc:
            failures.append(f"Cannot import {module_name}: {exc}")

    if not failures:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from detector import FastestDet
            model = FastestDet(Path(__file__).resolve().parents[1] / "models", threads=2)
            print("FastestDet: verified weights and loaded NCNN model")
            del model
        except Exception as exc:
            failures.append(f"Cannot load FastestDet NCNN model: {exc}")

    if args.picamera:
        try:
            from picamera2 import Picamera2
            cameras = Picamera2.global_camera_info()
            print(f"CSI cameras detected: {len(cameras)}")
            if not cameras:
                failures.append("No CSI camera found; check ribbon cable/camera configuration")
        except (ImportError, OSError, RuntimeError) as exc:
            failures.append(f"Cannot use Picamera2: {exc}")

    for message in failures:
        print(f"FAIL: {message}", file=sys.stderr)
    print("Ready for Pi browser demo" if not failures else "Preflight failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
