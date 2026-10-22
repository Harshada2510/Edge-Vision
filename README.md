# EdgeVision on Raspberry Pi 3 Model B (64-bit Raspberry Pi OS Lite)

This version processes each captured frame through **FastestDet (352×352, direct NCNN CPU)**, a small observation-only IoU tracker, IPM-based forward-warning logic, optional lane analysis and an OpenCV video overlay. The browser is a demo viewer of that annotated video; the same overlay can feed an in-car display. **IPM is on by default** with the estimated geometry discussed for the supplied Pexels video. The default profile prioritises vehicle detection and leaves lanes off. Video replay and live camera share the same processing path.

This is a development prototype, not a validated driving safety device. Physical Pi 3 Model B throughput and warning accuracy must be measured on the target camera and road conditions. Host-computer numbers cannot establish Pi FPS.

## Set up your Pi: 64-bit Raspberry Pi OS Lite (Debian Trixie)

This guide assumes your confirmed **Raspberry Pi 3 Model B, 64-bit Trixie-based Lite image** and your reported 905 MiB of RAM. The Pi runs the server without a desktop; the browser runs on your laptop. Download `EdgeVision_Pi3B_NCNN_pi3b_3threads_20260927.zip` and the accompanying `EdgeVision_demo_640x360_10fps.mp4` to your laptop. This demo clip was resized from the supplied 1280×720, 30 FPS Pexels video before transfer; the Pi decodes only 640×360 at 10 FPS while retaining the 16:9 IPM geometry:

```bash
# On your laptop (replace account and IP):
scp EdgeVision_Pi3B_NCNN_pi3b_3threads_20260927.zip YOUR_USER@YOUR_PI_IP:~/
scp EdgeVision_demo_640x360_10fps.mp4 YOUR_USER@YOUR_PI_IP:~/drive.mp4

# In an SSH shell on the Pi (Python is included in Raspberry Pi OS Lite):
cd ~
python3 -m zipfile -e EdgeVision_Pi3B_NCNN_pi3b_3threads_20260927.zip EdgeVision
cd EdgeVision
```

Install the small runtime in an environment that can see system packages. Trixie normally supplies Python 3.13, and the pinned NCNN release has a compatible AArch64 wheel. NumPy, OpenCV and Flask come from Trixie's packages. The `--no-deps` switch prevents the NCNN wheel from installing a second pip OpenCV and its other unused package extras; `--only-binary=:all:` prevents a source build:

```bash
sudo apt update
sudo apt install --no-install-recommends python3-venv python3-opencv python3-numpy python3-flask
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --no-deps --only-binary=:all: -r requirements-pi.txt
python scripts/pi_preflight.py
```

`pi_preflight.py` checks the architecture and OS codename, prints actual memory and dependency versions, and loads the checksum-verified detector. It should report `aarch64`, `trixie`, and a working model. If pip cannot find the NCNN wheel, note `python3 --version` and compare it with the [published wheels](https://pypi.org/project/ncnn/1.0.20260526/#files); there is no need to replace your operating system.

For a Pi CSI camera, also run `sudo apt install --no-install-recommends python3-picamera2` and `python scripts/pi_preflight.py --picamera`. This checks that libcamera sees the camera. The package is unnecessary for USB capture or video replay. The `--system-site-packages` environment makes Raspberry Pi OS's Picamera2 binding available to the app.

On an x86 development computer, install `requirements-dev.txt` first, then `python -m pip install --no-deps -r requirements-pi.txt` in its virtual environment. This prevents installing both pip OpenCV variants during desktop development. The Pi instructions above continue to use Trixie's own OpenCV and Flask. The runtime does not import Torch or Ultralytics. The included, checksum-verified models and their licenses are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

## Run

```bash
# Start with the copied video; make the cockpit reachable by Pi hostname:
python main.py --video ~/drive.mp4 --realtime --loop --threads 3 --host 0.0.0.0

# When your CSI or USB camera is connected, use one of these instead:
python main.py --picamera --threads 3 --host 0.0.0.0
python main.py --camera 0 --threads 3 --host 0.0.0.0

# CPU throughput test; times include capture, resizing, NCNN, tracker and HUD.
python main.py --video ~/drive.mp4 --no-web --warmup 20 \
  --max-frames 200 --metrics results/pi3b-no-web.json

# Include the cockpit's capped JPEG encoding and memory use in the timing run.
python scripts/benchmark_pi.py --video ~/drive.mp4 --threads 2 3 4 \
  --web --output results/pi3b-web-comparison.json
```

### Simplify the video overlay and find the FPS bottleneck

The default `--hud-style compact` draws thin vehicle rectangles, **one IPM distance badge for each detected vehicle**, optional lane boundaries, and a solid banner when an alert is active. It omits track IDs, confidence, class labels, radar, and performance bars. FCW, IPM, and telemetry calculations still run for every frame. A displayed `~` means the IPM distance uses estimated camera parameters; `?` means ground projection could not produce a usable distance for that vehicle. The browser BBOX toggle hides both boxes and badges. `--hud-style full` restores the detailed OpenCV video HUD. `--hud-style off` removes **all** video warnings for diagnostics only; it does not stop the underlying risk logic. The browser toggles can add radar/telemetry to compact mode, but these are off by default. No HTML layer is required for the final video overlay.

The earlier Pi screenshot showed about **89 ms of inference per frame and 135 ms for one completed frame**, with about **5 FPS observed**. Controlled replay on that same Pi 3B, with lanes off and web JPEG enabled, subsequently measured **9.72 FPS with compact HUD at 2 NCNN threads, 10.68 FPS at 3 threads, and 6.77 FPS at 4 threads**. The 3-thread trial measured median inference 74.0 ms, HUD 0.1 ms and JPEG 6.5 ms per encoded frame; its p95 complete-frame latency was 100.4 ms. These results support a **3-thread default on this setup**. They do not establish throughput with an attached camera, lane model, active browser clients, or a physical car display.

Stop the existing service for a clean comparison on the **actual Pi**, then run the same file, thread count and JPEG settings for all three styles:

```bash
sudo systemctl stop edgevision
python scripts/benchmark_pi.py --video ~/drive.mp4 --threads 2 \
  --hud-styles full compact off --web --output results/pi3b-hud-web.json
sudo systemctl start edgevision
```

The benchmark replays as fast as the Pi permits, without `--realtime`, and prints observed throughput and median inference, HUD and JPEG times plus p95 total time. Compare `full` with `compact` for the overlay benefit, then `compact` with `off` for its remaining drawing cost. The JSON keeps capture, resize, detector preparation/postprocessing, tracking, pipeline and total timings. To isolate browser JPEG cost, rerun without `--web` (when doing so, no video is displayed). The measured 3-thread setting is now the default; retest on the actual camera and with browser clients before treating 10 FPS as sustained. Rendering fewer graphics cannot overcome a model inference bottleneck by itself. `--no-web` is a throughput diagnostic until a physical display/video output is wired into the runner.

### Open the browser using your Pi hostname

With the Pi and laptop on the same reachable network, open `http://YOUR_PI_HOSTNAME:5000/` on your laptop. Use the hostname that actually resolves in your laptop's browser; if your network uses mDNS, try `http://YOUR_PI_HOSTNAME.local:5000/`. The hostname replaces the IP address, while `:5000` remains because the app listens on port 5000. No forwarding or desktop session on the Pi is needed.

Check locally on the Pi with `curl http://127.0.0.1:5000/api/telemetry`. If that responds but the laptop cannot connect, check that the command or service includes `--host 0.0.0.0`. A name configured only in `~/.ssh/config` might work for `ssh` yet not resolve in a browser; try the Pi's LAN address from `hostname -I` or configure name resolution. Keep the demo on a trusted local network: the current browser controls have no authentication.

When you are on another network, or intentionally keep the service bound to `127.0.0.1`, an SSH tunnel is an optional fallback. Start the server with `--host 127.0.0.1`, run this on the laptop, then visit `http://127.0.0.1:5000/`:

```bash
ssh -N -L 5000:127.0.0.1:5000 YOUR_USER@YOUR_PI_HOSTNAME
```

To keep the video demo running after you disconnect SSH, install [the systemd template](deploy/edgevision.service.example) with your actual SSH username. The template already listens on the LAN for hostname access:

```bash
sed "s/CHANGE_ME/$(id -un)/g" deploy/edgevision.service.example | sudo tee /etc/systemd/system/edgevision.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable edgevision
sudo systemctl restart edgevision
journalctl -u edgevision -f
```

The template expects the extracted application in `/home/YOUR_USER/EdgeVision` and the copied clip at `/home/YOUR_USER/drive.mp4`, as above. For CSI, edit the installed unit and use its commented `--picamera` `ExecStart` in place of the `--video` line, then run `sudo systemctl daemon-reload && sudo systemctl restart edgevision`. For an existing installation, transfer the updated ZIP to your Pi, run `python3 -m zipfile -e ~/EdgeVision_Pi3B_NCNN_pi3b_3threads_20260927.zip /path/to/your/EdgeVision` in SSH, then restart the service. This updates the code without reinstalling its virtual environment.

For your existing `~/edgevision_ncnn/v2` checkout, use `../v1/.venv/bin/python` from the `v2` directory; the virtual environment does not need to be copied. You can use 3 threads now without downloading updated code: `cd ~/edgevision_ncnn/v2 && ../v1/.venv/bin/python main.py --video ../assets/drive.mp4 --realtime --loop --threads 3 --hud-style compact --host 0.0.0.0`. The installed systemd service may still explicitly request `--threads 2`. Change its `ExecStart` to `--threads 3` (and `OMP_NUM_THREADS=3` if it sets that variable), run `sudo systemctl daemon-reload`, then restart the service. Updating Python's default alone cannot override an explicit service argument.

With your reported **118 MiB used of 905 MiB RAM** and unused swap, the tested defaults are 3 NCNN threads, lanes off, compact native overlay, 640 pixel processing width, JPEG quality 70 and stream at most 10 FPS. Captured frames overwrite a single slot; Picamera2 uses two camera buffers. Web clients share one JPEG stream, and timing samples keep at most 1,200 frames rather than growing through a long demo. Inference and FCW still run on each processed frame. The browser/API show process RSS, and the final report records peak RSS. `free -m`, `top` and `vcgencmd get_throttled` show real memory pressure and CPU throttling on the Pi. The service template sets a 700 MiB memory ceiling. Torch, TensorFlow and Ultralytics are not needed at runtime.

`--realtime` skips video frames that have become stale; risk engines use the recorded frame times. Without it, replay processes each frame sequentially and measures processing throughput regardless of the video's playback rate. A live camera has a single latest-frame slot, so slow inference cannot build up a queue of old frames. `--max-frames` exits for tests; omit it to keep the live cockpit running. `--save-frame result.jpg` writes the last annotated frame.

The report distinguishes model inference, preparation, tracking, HUD, JPEG and full frame timings, excludes the requested warmup frames, and records the host/board. Runs of up to 1,200 timed frames retain complete timing statistics; for longer runs, median, p95 and `observed_fps` cover the latest 1,200 frames and `timed_frames` still counts all frames. Full frame timing includes waiting for a fresh camera frame (or reading video); a live camera's separate acquisition thread overlaps model processing. Check median **and** p95, temperature and throttling on the physical Pi. If the web service is running, stop it before running separate CPU benchmarks to avoid two processes contending for the 905 MiB RAM. Tune `--threads 2`, `3`, `4` and, only if required, `--process-width`; detection input stays 352×352. A few FPS on either side of 10 is an engineering target, not a measured guarantee.

## Calibrate metric IPM

The default [estimated camera profile](config/estimated_camera.json) uses reference size 1280×720, pinhole `fx=fy=800 px`, centre `(640,360)`, assumed 1.5 m height and 1.4° down pitch derived from a vanishing point near `(640,341)`. Pixel focal length and camera height are **assumptions**, and rolling/cropping to a different camera can change everything. IPM intersects the projected box bottom centre with the flat ground plane, giving an *estimated* forward/lateral position (marked `~` in HUD and browser). FCW uses this forward range and its change across confirmed observations to calculate closing speed and TTC. It suppresses metric alerts if projection is missing or outside the usable region. The browser/API say `quality: estimated` and the benchmark says `calibrated: false`. **These warnings are for demonstration, not trusted collision prevention.**

To improve reliability, measure the mounted Pi camera instead of using the Pexels estimate:

1. Mount and lock the camera. If the lens has material distortion, obtain its intrinsic matrix and distortion coefficients (for example by chessboard calibration); the IPM code undistorts each contact point before projection. Set `lens_corrected: true` **only** if the input image is already corrected. Do not change crop, resolution aspect ratio, zoom, mount or lens after calibration.
2. Put four **measured points on the same flat road plane** in front of the vehicle. Mark their centres in this order: far left, far right, near right, near left. All ground coordinates are `[right_m, forward_m]`, measured from the `origin` stated in the JSON. Positive `right_m` means vehicle right; positive `forward_m` means ahead.
3. Record a frame with the **actual mounted Pi camera**, using its final lens and resolution. The Pexels demo clip cannot supply physically measured ground coordinates for that camera. To work from a camera recording, use `python scripts/calibration_points.py --video ~/YOUR_CAMERA_RECORDING.mp4 --frame 100 --output config/calibration.json --export-frame frame.png`, transfer that raw frame to your laptop over SCP, and enter its image pixel coordinates in a copy of [the example](config/calibration.example.json). The export option saves the raw frame and does not generate a completed calibration JSON.
4. Fill `ground_points_m`, `origin`, `image_size`, and lens calibration. Then set `measured: true`. Launch with `--calibration config/calibration.json`. The program rejects mismatched aspect ratios, degenerate quads and missing lens declarations rather than inventing a distance.
5. Verify the predicted distances against held-out measured road marks, across the near and far operating range, before using numbers in a dashboard. Range is only shown within the measured quadrilateral, with the object's **box bottom centre** treated as the road contact point. This assumption fails for occluded, truncated or elevated objects.

The file `config/calibration.example.json` is intentionally incomplete and `measured: false`; it must **not** be used as a working calibration. See `tests/test_ipm_distance.py` for a self-contained synthetic example with known coordinates.

## Ego speed, lanes and model tradeoffs

FCDW runs only while the Pi receives **fresh external vehicle-speed evidence**. A producer can atomically write a JSON file containing `{"unix_time": 1790450000.0, "speed_mps": 0.0}` (replace the timestamp with the producer's actual UTC epoch time), updating at least once per second. Start `python main.py --camera 0 --ego-speed-file config/ego_speed.json`. Missing, stale, invalid, or moving readings reset FCDW and mark its status `NO SPEED`. Video replay has no matching speed log by default; it does not assume a stopped vehicle.

Optional `--lanes trilite` loads the small TriLiteNet Tiny 320×192 NCNN lane head. It processes a frame about every 0.35 seconds by default; use `--lane-period` to change that. Alerts require two fresh coherent boundaries; gaps expire in 0.6 seconds. This converted model has not been validated on the supplied city's road markings and will lower Pi throughput. `--lanes hough` is a conservative, optional classical baseline and is sensitive to shadows, road texture and bonnet edges. Leave lanes off for the initial 10 FPS run.

For visual inspection of TriLite on the Pi, enable the **OpenCV video** debug overlay:

```bash
cd ~/edgevision_ncnn/v2
../v1/.venv/bin/python main.py --video ../assets/drive.mp4 --realtime --loop \
  --threads 3 --lanes trilite --lane-overlay --hud-style compact --host 0.0.0.0
```

Open the cockpit at `http://YOUR_PI_HOSTNAME:5000/`. The stream now tints the raw segmentation pixels magenta, outlines the sampling region in cyan, draws accepted lane boundaries in green (magenta when departing), and reports `PAIR OK` or `NO PAIR` with the prediction's age in a small upper-left badge below the alert banner. `NO PAIR` still shows raw pixels, so you can distinguish a bad model prediction from a rejected geometric fit. Results older than 0.6 seconds are hidden; disabling the AR LANES video layer hides the debug graphics. This overlay is optional and adds drawing cost: omit `--lane-overlay` for normal running. To measure that cost on the Pi, run `scripts/benchmark_pi.py` with `--lanes trilite --lane-overlay --hud-styles compact --web` and compare with the same command without `--lane-overlay`. Check `vcgencmd get_throttled` during both trials; an active undervoltage flag invalidates their FPS comparison.

After extracting an updated package into `~/edgevision_ncnn/v2`, run `../v1/.venv/bin/python main.py --version`. This package prints `EdgeVision pi3b-lane-distance-20260927-r4`. If the option is missing or prints an older build, `v2/main.py` has not been replaced with the current package.

Large, wide boxes near the bottom of some dashcam frames are the camera vehicle's bonnet. The default suppression begins when box bottom is past 90% of frame height and box width exceeds 35%. Review `--save-frame` for your camera; `--bonnet-bottom` adjusts that vertical fraction (up to `1.0` to disable the default test). Nearby real cars can also touch that part of the image, so this heuristic needs camera-specific validation.

The model detects COCO bicycle, car, motorcycle, bus and truck classes only. A stationary vehicle at estimated short ground range can produce CAUTION, while WARNING/DANGER require a confirmed track with sustained closing evidence. The estimates do **not** account for ego steering, braking, camera-to-bumper offset, road slope, occlusion, object depth ambiguity or camera vibration.

## Tests

```bash
python tests/run_all_tests.py
```

The nine suites exercise model loading and a short actual NCNN replay, homography math, missing speed evidence, tracker observation gaps, lane expiry, the web API and HUD. The 10 FPS requirement still needs a run on the **actual Pi 3 Model B** with the mounted camera, a representative video and the chosen web/lane settings.
