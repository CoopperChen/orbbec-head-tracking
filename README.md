# Orbbec Head Tracking

Production-oriented 6-DoF head tracking for a single Orbbec Gemini 2L depth sensor.

## Setup

Create a Python 3.11 virtual environment, then install this package:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

This project pins Orbbec's official Windows Python 3.11 wheel because the older PyPI `pyorbbecsdk` wheel can install non-Windows binaries. It also pins `mediapipe==0.10.14` because newer MediaPipe wheels may expose only the Tasks API and omit the classic FaceMesh solution used by the tracker.

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

- Acquire synchronized color and depth frames through `pyorbbecsdk.Pipeline`.
- Align depth to color with `AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)`.
- Copy raw SDK frame buffers before NumPy/OpenCV decoding.
- Scale `Y16` depth frames with `depth_frame.get_depth_scale()` into `float32` millimeters.
- Extract MediaPipe FaceMesh landmarks.
- Estimate pose with one of:
  - `depth-rigid`: depth-assisted rigid fit (with distortion-consistent depth rays).
  - `pnp`: `cv2.solvePnPRansac` on 2D landmarks.
  - `hybrid`: depth-assisted rigid fit, then `cv2.solvePnPRansac` refinement.
