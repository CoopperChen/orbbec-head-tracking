# Orbbec Head Tracking

Production-oriented 6-DoF head tracking for a single Orbbec Gemini 2L depth sensor, with optional real-time HICON XYZBC compensation over UDP for 5-axis CNC.

## Setup

Stack pinned for Gemini 2L on Windows:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 (64-bit) | |
| `pyorbbecsdk2` | ≥ 2.0.18 | Import as `from pyorbbecsdk import ...` |
| `numpy` | ≥ 1.24, &lt; 2 | MediaPipe + Orbbec wheels |
| `opencv-python` | ≥ 4.10, &lt; 4.13 | Avoid 4.13+ (NumPy 2) |
| `mediapipe` | == 0.10.14 | Classic FaceMesh API |

Do not install `opencv-contrib-python` alongside `opencv-python`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# CNC streaming (YAML calibration + Mach4 work pose)
pip install -e ".[cnc]"

# Development / tests
pip install -e ".[dev,cnc]"
```

Verify the package and camera (close Orbbec Viewer first):

```powershell
python -c "import orbbec_head_tracking.orbbec_sdk; print('package ok')"
python scripts\check_orbbec_device.py
```

On first use on Windows 10, run Orbbec’s metadata script as Administrator ([pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk) `scripts/obsensor_metadata_win10.ps1`), then reboot. If Python reports `count 0` but Orbbec Viewer works, install [OrbbecSDK v2](https://github.com/orbbec/OrbbecSDK_v2/releases) and use `pyorbbecsdk2` ≥ 2.1.1.

## Commands

| Command | Module | Purpose |
|---------|--------|---------|
| `orbbec-head-tracker` | `tracker.py` | Print pose to stdout (mm + pitch/yaw/roll) |
| `orbbec-head-viewer` | `tracker.py` | Live RGB + depth windows with pose overlay |
| `orbbec-head-stream-cnc` | `stream_cnc.py` | Head tracking → HICON UDP XYZBC user offsets |
| `orbbec-cnc-pipeline-benchmark` | `benchmark_cnc_pipeline.py` | Per-stage latency benchmark + raw CSV export |
| `orbbec-cnc-offset-test` | `cnc_offset_test.py` | Manual offset sliders for controller bring-up |

### Head tracking

```powershell
orbbec-head-tracker
orbbec-head-viewer
orbbec-head-tracker --view
orbbec-head-viewer --no-smoothing
orbbec-head-viewer --pose-solver pnp
orbbec-head-tracker --offline-npz "path\to\frames.npz"
```

Press `q` or `Esc` in viewer windows. Smoothing defaults can be tuned with `--translation-alpha`, `--rotation-alpha`, and `--translation-deadband-mm`.

Offline `.npz` must include `color_bgr`, `depth_mm`, `camera_matrix`, and `distortion_coefficients` (optional `ts`).

### CNC compensation (UDP)

```powershell
orbbec-head-stream-cnc `
  --work-pose-udp-port 62100 `
  --capture-baseline-sec 2 `
  --view
```

`--calibration` defaults to `config/cnc_compensation_example.yaml` (cwd, else repo root). Override with `--calibration path\to\file.yaml`.

For quieter motion (less catch-up thrash when the vision loop is slow), use the clinical profile:

```powershell
orbbec-head-stream-cnc --calibration config/cnc_compensation_quiet.yaml --log
```

That profile disables catch-up bursts, caps each packet at 0.25 mm / 0.15 mm (XY/Z), halves vmax, and widens the offset deadband. Expect more lag behind a moving head; check the log for `catch_up` near 0% and `step_mm` mostly ≤ 0.25.

- **Follow** mode (default): offsets move the machine with the head so the nozzle stays on the scalp trace.
- Default HICON UDP: controller `192.168.208.35`, local bind `192.168.208.10` (`--device-ip` / `--bind-ip` to override).
- On tracking loss, spike rejection, or link fault: **hold last offset** (not a blind zero flash). See `safety` in the YAML.
- `--no-ack-watchdog` for view/dry-run without a connected controller.
- Static pose fallback: `--machine-pose=X,Y,Z,B,C` or `machine_pose` in YAML.

**Mach4 active work coordinates:** publish G54 DRO values with [`scripts/mach4_work_pose_publisher.lua`](scripts/mach4_work_pose_publisher.lua) (multi-target UDP: Orbbec `62100`, layout_design `record-pm` `62101`). Setup: [`docs/mach4-work-pose-udp.md`](docs/mach4-work-pose-udp.md).

**Offset bring-up without camera:**

```powershell
orbbec-cnc-offset-test --device-ip 192.168.208.35 --bind-ip 192.168.208.10
```

### Long-run stability test

Hold a head steady in front of the camera and stream for two hours while logging head pose and commanded XYZBC offsets to one CSV, then check whether the offsets drift.

```powershell
orbbec-head-stream-cnc `
  --duration-sec 7200 `
  --log --log-rate-hz 10 `
  --capture-baseline-sec 5 `
  --no-ack-watchdog
```

`--log` on its own writes `results/cnc_stream_<YYYYMMDD_HHMMSS>.csv`; pass a directory (`--log D:\runs`) or a file path (`--log results\stability.csv`) to place it elsewhere — the timestamp is always appended to the stem. Use `--log-csv path.csv` instead when you need an exact, non-timestamped filename (add `--log-timestamped` to stamp it).

Add `--work-pose-udp-port 62100` (or `--machine-pose`) so B/C encoding uses the real nozzle pose; drop `--no-ack-watchdog` when the controller is connected. The run stops on its own after `--duration-sec`, and Ctrl+C does the same. Rows are flushed every ~5 s, so an interrupted run still leaves a usable log.

**Shutdown never moves the machine.** A user offset keeps the tool tracking the head, so zeroing it on exit would drag the tool by the full standing offset relative to the head. Instead the offset is left in the controller and reported on stdout — as a banner above `EXIT_OFFSET_WARN_MM` (5 mm). Clear it in Mach4 once the tool is clear of the workpiece: the tracker cannot read the controller's current offset, so it assumes zero at startup and an uncleared offset becomes a single-step jump on the next run.

`--log-rate-hz` decimates the control loop, which `update_period_ms: 10` paces at a nominal 100 Hz but which in practice runs at the camera rate (~30 Hz — see the latency benchmark below). Columns per sample: `t_s`, `wall_clock`, head pose (`head_x_mm`…`roll_deg`, `rvec_*`), head Δ in machine frame (`head_dx_mm`…), the encoder's required offset (`req_x`…`req_c`), the offset actually sent after mismatch and safety (`off_x`…`off_c`), plus `confidence`, `head_speed_mm_s`, `loop_dt_ms`, `step_mm`, `step_deg`, `tracking_ok`, `baseline_ready`, `link_ok`, and the safety `action`/`reason`.

`loop_dt_ms` is the measured tick, and `step_mm` / `step_deg` are how far a single packet moved the offset — the controller applies a user offset inside one servo cycle, so step size is what the drive actually feels, not average velocity. `--step-warn-mm` / `--step-warn-deg` warn when a packet exceeds a threshold, and the largest step of the run is printed at shutdown.

Then plot every channel against time:

```powershell
python scripts\figures\plot_stability.py --log results\cnc_stream_20260730_090000.csv
```

This writes `results/figures/<log stem>_drift.{pdf,svg,png}` (four stacked panels: CNC XYZ, CNC B/C, head ΔXYZ, head Δpitch/yaw/roll — per-bin mean with a min/max jitter envelope, tracking dropouts shaded) and `results/<log stem>_drift_summary.csv` with mean, std, peak-to-peak, and the least-squares **drift per hour** for each channel. The same drift table is printed to stdout, and each per-hour slope is repeated in the panel legends.

### Pipeline latency benchmark

Measure per-stage processing time with `time.perf_counter()` and export raw CSV for statistical analysis (median, P95, etc.). Runs the full CNC loop at a lower control rate (default 30 Hz) so each stage can be timed without overrunning the period.

**Activate the venv first** — CLI scripts are installed under `.venv\Scripts\` when you run `pip install -e ".[cnc,dev]"`. After adding new entry points, reinstall once:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[cnc,dev]"
```

```powershell
orbbec-cnc-pipeline-benchmark `
  --calibration config/cnc_compensation_example.yaml `
  --loops 300 --warmup 30 `
  --output results/pipeline_timing.csv `
  --summary results/pipeline_timing_summary.csv `
  --verbose
```

Per-loop CSV columns:

| Column | Stage |
|--------|--------|
| `frame_acquire_align_ms` | Orbbec wait + depth-to-color align + decode |
| `landmark_ms` | MediaPipe FaceMesh |
| `pose_estimate_ms` | Depth-rigid 6-DoF + stabilization |
| `temporal_filter_ms` | Pose smoother |
| `encode_ms` | XYZBC offset encoding |
| `safety_rate_limit_ms` | Mismatch targeting + safety guards |
| `udp_ms` | UDP pack/send (+ ACK poll) |
| `loop_total_ms` | Full loop processing time |
| `loop_period_ms` | Wall time between loop starts |

All loops are written to CSV (including warmup). The stdout summary excludes the first `--warmup` samples (default 30). Run **without** `--view` for pure processing latency. ACK watchdog is off by default (`--ack-watchdog` to enable).

Without activating the venv:

```powershell
& ".\.venv\Scripts\orbbec-cnc-pipeline-benchmark.exe" --calibration config/cnc_compensation_example.yaml
```

## Project layout

```
config/
  cnc_compensation_example.yaml   # CNC calibration, limits, safety, motor map
  cnc_compensation_quiet.yaml     # Same calibration; quieter safety (no catch-up thrash)
docs/
  cnc-udp-pipeline.md           # Production UDP pipeline diagram
  face-tracking-pipeline.md     # Vision-only pipeline diagram
  mach4-work-pose-udp.md        # Mach4 → Orbbec work-coordinate bridge
scripts/
  check_orbbec_device.py        # Quick camera enumeration
  mach4_work_pose_publisher.lua # Mach4 PLC Lua publisher
  pipeline_demos/               # Per-stage OpenCV viewers (RGB, depth, pose, …)
src/orbbec_head_tracking/       # Main package (see modules below)
tests/                          # pytest suite
```

## Package modules

All modules live under `src/orbbec_head_tracking/`.

### Vision pipeline

| Module | Role |
|--------|------|
| [`tracker.py`](src/orbbec_head_tracking/tracker.py) | `OrbbecHeadTracker`: capture, align, FaceMesh, pose solvers, CLI (`main`, `viewer_main`) |
| [`orbbec_sdk.py`](src/orbbec_head_tracking/orbbec_sdk.py) | Re-exports `pyorbbecsdk`; Windows DLL path registration |
| [`frames.py`](src/orbbec_head_tracking/frames.py) | Color/depth frame decode, intrinsics from stream profile |
| [`geometry.py`](src/orbbec_head_tracking/geometry.py) | Depth sampling, weighted rigid fit, rotation utilities, Euler helpers |
| [`smoothing.py`](src/orbbec_head_tracking/smoothing.py) | `PoseSmoother`: translation EMA + rotation SLERP with deadbands |
| [`config.py`](src/orbbec_head_tracking/config.py) | `TrackerConfig`: solver, smoothing, PnP, depth-rigid parameters |
| [`constants.py`](src/orbbec_head_tracking/constants.py) | Face model 3D points, landmark indices, axis model for viz |
| [`types.py`](src/orbbec_head_tracking/types.py) | `HeadPose`, `TrackingFrame` datatypes |
| [`viz.py`](src/orbbec_head_tracking/viz.py) | `draw_pose_overlay`: landmarks, axes, pose readout on RGB |

Pose solvers (`--pose-solver`):

- **`depth-rigid`** (default): back-project landmarks with aligned depth, rigid fit to face model.
- **`pnp`**: `cv2.solvePnPRansac` on 2D landmarks + approximate model.
- **`hybrid`**: depth-rigid initial guess, then PnP refinement.

### CNC compensation pipeline

| Module | Role |
|--------|------|
| [`stream_cnc.py`](src/orbbec_head_tracking/stream_cnc.py) | 100 Hz loop: tracker → encode → safety → UDP; CLI entry point |
| [`benchmark_cnc_pipeline.py`](src/orbbec_head_tracking/benchmark_cnc_pipeline.py) | Per-stage latency benchmark; CSV export for paper stats |
| [`pipeline_timing.py`](src/orbbec_head_tracking/pipeline_timing.py) | `LoopTimingSample`, CSV writer, median/P95 summary helpers |
| [`cnc_config.py`](src/orbbec_head_tracking/cnc_config.py) | `CncCompensationConfig`, YAML loader, axis limits, safety/mismatch/deadband |
| [`cnc_offset_encoder.py`](src/orbbec_head_tracking/cnc_offset_encoder.py) | Baseline capture, head Δ → `CncUserOffset` XYZBC; B/C modes; zero latch |
| [`cnc_kinematics.py`](src/orbbec_head_tracking/cnc_kinematics.py) | 5-axis FK, tool normal, pose-aware B/C IK, XYZ tip correction |
| [`cnc_safety.py`](src/orbbec_head_tracking/cnc_safety.py) | Rate limits, spike reject, hold-last, recovery window after loss |
| [`cnc_stability_log.py`](src/orbbec_head_tracking/cnc_stability_log.py) | Long-run CSV logger; drift slope / spread analysis helpers |
| [`cnc_mismatch.py`](src/orbbec_head_tracking/cnc_mismatch.py) | Required vs sent offset tracking; optional snap / integral correction |
| [`cnc_protocol.py`](src/orbbec_head_tracking/cnc_protocol.py) | `MSG_SET_AXIS_USEROFFSET` pack/unpack (48-byte UDP), motor map |
| [`cnc_udp_streamer.py`](src/orbbec_head_tracking/cnc_udp_streamer.py) | UDP bind/send, ACK watchdog, link status |
| [`cnc_work_pose_client.py`](src/orbbec_head_tracking/cnc_work_pose_client.py) | Receive Mach4 active work-coordinate JSON over UDP |
| [`cnc_viz.py`](src/orbbec_head_tracking/cnc_viz.py) | CNC status panel overlay (offsets, link, safety, work pose) |
| [`cnc_offset_test.py`](src/orbbec_head_tracking/cnc_offset_test.py) | Interactive OpenCV slider UI for manual offset testing |

### Data flow (CNC)

```
Orbbec → tracker → smoothing → CncOffsetEncoder (vs baseline + live work pose)
       → CncMismatchTracker → CncSafetyGuards → UserOffsetMessage → CncUdpStreamer → HICON
```

Diagram: [`docs/cnc-udp-pipeline.md`](docs/cnc-udp-pipeline.md) · Vision-only: [`docs/face-tracking-pipeline.md`](docs/face-tracking-pipeline.md)

## Configuration

[`config/cnc_compensation_example.yaml`](config/cnc_compensation_example.yaml) — key sections:

| Section | Purpose |
|---------|---------|
| `machine` | Link lengths `a_mm`, `d_mm`, gap geometry for FK |
| `camera_extrinsic` | 3×3 camera → machine rotation |
| `bc_mode` | `tool_normal_ik` or `camera_rvec` for B/C encoding |
| `machine_pose` | Fallback nozzle XYZBC (work coords) |
| `work_pose_udp` | Listen for live Mach4 work pose (port 62100) |
| `motor_map` | Logical XYZBC → HICON motor indices 0–7 |
| `axis_limits` | Per-axis compensation clamps |
| `safety` | Confidence gate, rate limits, hold-last, recovery ticks |
| `mismatch` | Tracking-error correction (`kp`/`ki`, snap) |
| `offset_deadband` | Hysteresis near zero |

## Scripts

| Path | Purpose |
|------|---------|
| [`scripts/check_orbbec_device.py`](scripts/check_orbbec_device.py) | List connected Orbbec devices |
| [`scripts/mach4_work_pose_publisher.lua`](scripts/mach4_work_pose_publisher.lua) | Mach4 `mc.mcAxisGetPos` → UDP JSON |
| [`scripts/pipeline_demos/`](scripts/pipeline_demos/) | Stage-by-stage viewers: RGB, depth, align, landmarks, pose (`demo_all.py`) |
| [`scripts/figures/plot_stability.py`](scripts/figures/plot_stability.py) | Drift plot + summary CSV from a `--log` stability run |

## Tests

```powershell
python -m pytest tests/ -q --ignore=tests/test_tracker_integration.py
```

| Test file | Covers |
|-----------|--------|
| `test_geometry.py` | Rigid fit, depth sampling, rotation math |
| `test_smoothing.py` | Pose smoother deadbands and SLERP |
| `test_cnc_protocol.py` | UDP message pack/unpack golden bytes |
| `test_cnc_config.py` | YAML calibration loading |
| `test_cnc_kinematics.py` | FK, B/C IK, pose-aware solver |
| `test_cnc_offset_encoder.py` | Baseline encode, B/C modes, limits |
| `test_cnc_bc_euler.py` | Euler → B/C mapping |
| `test_cnc_mismatch.py` | Mismatch tracker, preserve-sent |
| `test_cnc_safety_catchup.py` | Rate limit, catch-up, spike reject |
| `test_cnc_hold_zero_drift.py` | Hold-last and recovery after loss |
| `test_cnc_zero_latch.py` | Per-axis zero hysteresis |
| `test_cnc_work_pose_client.py` | Mach4 work-pose UDP parser |
| `test_cnc_offset_test.py` | Offset test CLI helpers |
| `test_cnc_stability_log.py` | Stability CSV writer, rate limiting, drift statistics |
| `test_pipeline_timing.py` | Benchmark CSV writer and summary stats |

## Architecture notes

**Vision**

1. Synchronized color + depth via `pyorbbecsdk2` `Pipeline`.
2. Hardware depth-to-color align (`AlignFilter`).
3. Defensive `.copy()` on SDK buffers before decode.
4. Depth scaled to mm with `get_depth_scale()`.
5. MediaPipe FaceMesh on RGB.
6. 6-DoF pose (depth-rigid / PnP / hybrid).
7. Optional temporal smoothing.

**CNC**

1. Capture head baseline for ~2 s at print start.
2. Each tick: head Δ vs baseline → proposed XYZBC offset.
3. Live nozzle pose from Mach4 work UDP (or YAML fallback) for pose-aware B/C.
4. Mismatch layer reconciles required vs sent offset.
5. Safety: rate limits, spike filter, hold on loss, recovery ramp.
6. Stream `MSG_SET_AXIS_USEROFFSET` to HICON at 100 Hz.

Native Orbbec SDK and MediaPipe stderr is suppressed by default; use `--verbose` on tracker/CNC CLIs to show it.
