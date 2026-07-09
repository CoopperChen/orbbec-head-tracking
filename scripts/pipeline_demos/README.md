# Pipeline stage demos (temporary)

Small OpenCV viewers for each step in the head-tracking pipeline. Requires the Orbbec Gemini 2L connected, Python 3.11+, and project dependencies installed.

```powershell
cd "D:\Research\Orbbec CV - cursor"
.\.venv\Scripts\Activate.ps1
pip install -e .
cd scripts\pipeline_demos
python demo_all.py
```

Demos import the SDK via `orbbec_head_tracking.orbbec_sdk` (install `pyorbbecsdk2`; same `pyorbbecsdk` import name).

| Script | Shows |
|--------|--------|
| `demo_rgb.py` | RGB frames |
| `demo_depth.py` | Native depth (before alignment) |
| `demo_landmarks.py` | MediaPipe face mesh on RGB |
| `demo_align.py` | RGB \| aligned depth \| overlay (same resolution) |
| `demo_pose.py` | Pose solver: 6 anchors, depth mm labels, RGB axes, X/Y/Z + pitch/yaw/roll |
| `demo_all.py` | **All stages at once** — 2×3 grid: RGB, depth, landmarks, align, aligned depth, pose |

Press **Q** or **Esc** to quit any viewer.

```powershell
python demo_rgb.py
python demo_depth.py
python demo_landmarks.py
python demo_align.py
python demo_pose.py
python demo_all.py
```

`demo_pose.py` and `demo_all.py` use pose logic from `src/orbbec_head_tracking/` (on `sys.path` automatically). Smoothing is off so you see raw solver output.

```powershell
python demo_all.py
python demo_all.py --pose-solver pnp
python demo_pose.py --pose-solver depth-rigid
```

`demo_all.py` uses one camera session and updates every panel synchronously from the same frame.
