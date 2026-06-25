# Orbbec Head Tracking

Production-oriented 6-DoF head tracking for a single Orbbec Gemini 2L depth sensor.

## Setup

This project pins a compatible stack for Gemini 2 L on Windows:

- **Python 3.11** (64-bit)
- **`pyorbbecsdk2`** ≥ 2.0.18 (PyPI; import as `from pyorbbecsdk import ...`)
- **`numpy`** ≥ 1.24, &lt; 2 (MediaPipe FaceMesh + Orbbec prebuilt wheels)
- **`opencv-python`** ≥ 4.10, &lt; 4.13 (avoid NumPy 2-only OpenCV 4.13+)
- **`mediapipe==0.10.14`** (classic FaceMesh API used by the tracker)

Do not install `opencv-contrib-python` alongside `opencv-python`. Avoid `jax` 0.10+ (pulls NumPy 2).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Verify the package and camera (close Orbbec Viewer first):

```powershell
python -c "import orbbec_head_tracking.orbbec_sdk; print('package ok')"
python scripts\check_orbbec_device.py
```

If `orbbec_head_tracking.orbbec_sdk` is missing, you are not in the repo with `pip install -e .` applied — run `git pull` then `pip install -e .` again from the project root.

Direct SDK check (without the project wrapper):

```powershell
python -c "from pyorbbecsdk import Context; print(Context().query_devices().get_count())"
```

On first use on Windows 10, run Orbbec’s metadata script as Administrator (from the [pyorbbecsdk repo](https://github.com/orbbec/pyorbbecsdk) `scripts/obsensor_metadata_win10.ps1`), then reboot. If Python sees `count 0` but Orbbec Viewer works, install [OrbbecSDK v2](https://github.com/orbbec/OrbbecSDK_v2/releases) and use `pyorbbecsdk2` ≥ 2.1.1.

## Run

```powershell
orbbec-head-tracker
```

The tracker prints translation in millimeters and Euler orientation as pitch, yaw, and roll in degrees. Stop it with `Ctrl+C`.

To show the live RGB visualization with facial anchor points, projected head-pose axes, a pose readout, and a separate aligned depth stream window:

```powershell
orbbec-head-viewer
```

You can also launch the same view from the tracker command:

```powershell
orbbec-head-tracker --view
```

In the visualization window, press `q` or `Esc` to close it.

Pose smoothing is enabled by default to reduce small X/Z and rotation jitter. For a steadier display when the head is mostly still, lower the alpha values:

```powershell
orbbec-head-viewer --translation-alpha 0.12 --rotation-alpha 0.18 --translation-deadband-mm 4
```

For a more responsive display, raise them:

```powershell
orbbec-head-viewer --translation-alpha 0.45 --rotation-alpha 0.5 --translation-deadband-mm 1
```

To compare against raw PnP output:

```powershell
orbbec-head-viewer --no-smoothing
```

You can stream machine-readable pose updates with:

```powershell
orbbec-head-tracker --jsonl
```

To stream translation + Euler rotation over Ethernet (TCP) as JSONL:

```powershell
orbbec-head-stream-tcp --tcp-host "<receiver-ip-or-hostname>" --tcp-port 5005
```

To stream live head-motion compensation to a HICON 5-axis controller (XYZBC user offsets over UDP):

```powershell
pip install -e ".[cnc]"
orbbec-head-stream-cnc `
  --calibration config/cnc_compensation_example.yaml `
  --machine-pose=-60,0,40,0,0 `
  --capture-baseline-sec 2 `
  --view
```

Add `--view` to show live RGB + depth windows with pose axes and a CNC status panel (XYZBC offsets, UDP link, baseline, safety). Press `q` or `Esc` to stop.

The CNC stream uses **follow** mode by default: offsets move the machine with the head so the nozzle stays on the scalp trace. Default UDP targets are controller `192.168.208.35` and local bind `192.168.208.10` (override with `--device-ip` / `--bind-ip`). Safety guards zero all axes on tracking loss, low confidence, or UDP link fault. Edit `config/cnc_compensation_example.yaml` for axis limits, camera-to-machine rotation, and machine geometry (`a_mm`, `d_mm` from `layout_design`).

You can also run offline on saved frames:

```powershell
orbbec-head-tracker --offline-npz "path\\to\\frames.npz"
```

The `.npz` file must include `color_bgr`, `depth_mm`, `camera_matrix`, and `distortion_coefficients` arrays (and may include `ts` timestamps).

The default pose solver uses the aligned depth stream to back-project FaceMesh anchors into camera-space 3D, then fits a rigid transform to the face model. This avoids asking PnP to infer all 3D geometry from 2D landmarks and an approximate face model. To compare against the 2D PnP solver:

```powershell
orbbec-head-viewer --pose-solver pnp
```

Native Orbbec SDK and MediaPipe startup warnings are suppressed by default. To show them while debugging:

```powershell
orbbec-head-tracker --verbose
```

## Architecture

- Acquire synchronized color and depth frames through `pyorbbecsdk2` (`from pyorbbecsdk import Pipeline`).
- Align depth to color with `AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)`.
- Copy raw SDK frame buffers before NumPy/OpenCV decoding.
- Scale `Y16` depth frames with `depth_frame.get_depth_scale()` into `float32` millimeters.
- Extract MediaPipe FaceMesh landmarks.
- Estimate pose with one of:
  - `depth-rigid`: depth-assisted rigid fit (with distortion-consistent depth rays).
  - `pnp`: `cv2.solvePnPRansac` on 2D landmarks.
  - `hybrid`: depth-assisted rigid fit, then `cv2.solvePnPRansac` refinement.
