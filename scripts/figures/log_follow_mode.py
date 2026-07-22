"""Follow-mode test / logging harness for Figure 5b.

Records, per control tick, the tracked head pose and the commanded XYZBC user
offset produced by the *real* CncOffsetEncoder, to a CSV that fig5_followmode.py
plots. Two sources:

  --source live       Orbbec camera + head tracker (run this during a session).
  --source synthetic  Drive the real encoder with a scripted head trajectory
                      (no hardware; useful to draft the figure now).

Live example:
  python scripts/figures/log_follow_mode.py --source live \
      --calibration config/cnc_compensation_example.yaml \
      --duration 20 --capture-baseline-sec 2

Synthetic example:
  python scripts/figures/log_follow_mode.py --source synthetic --duration 12

Out: results/follow_mode_log.csv  (override with --output)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from orbbec_head_tracking.cnc_config import CncCompensationConfig, load_compensation_config
from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder
from orbbec_head_tracking.types import HeadPose

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CALIB = _REPO / "config" / "cnc_compensation_example.yaml"
_DEFAULT_OUT = _REPO / "results" / "follow_mode_log.csv"

FIELDNAMES = [
    "t_s",
    "head_x_mm", "head_y_mm", "head_z_mm",
    "pitch_deg", "yaw_deg", "roll_deg",
    "head_dx_mm", "head_dy_mm", "head_dz_mm",
    "off_x", "off_y", "off_z", "off_b", "off_c",
    "confidence", "tracking_ok",
]


def _load_config(calibration: str | None) -> CncCompensationConfig:
    path = calibration or (str(_DEFAULT_CALIB) if _DEFAULT_CALIB.exists() else None)
    if path is None:
        return CncCompensationConfig()
    return load_compensation_config(path)


def _row(t, head, euler, delta, off, conf, ok) -> dict:
    dx, dy, dz = (delta if delta is not None else (0.0, 0.0, 0.0))
    return {
        "t_s": round(float(t), 4),
        "head_x_mm": round(float(head[0]), 3),
        "head_y_mm": round(float(head[1]), 3),
        "head_z_mm": round(float(head[2]), 3),
        "pitch_deg": round(float(euler[0]), 3),
        "yaw_deg": round(float(euler[1]), 3),
        "roll_deg": round(float(euler[2]), 3),
        "head_dx_mm": round(float(dx), 3),
        "head_dy_mm": round(float(dy), 3),
        "head_dz_mm": round(float(dz), 3),
        "off_x": round(float(off.x), 4),
        "off_y": round(float(off.y), 4),
        "off_z": round(float(off.z), 4),
        "off_b": round(float(off.b), 4),
        "off_c": round(float(off.c), 4),
        "confidence": round(float(conf), 3),
        "tracking_ok": int(bool(ok)),
    }


def _synthetic_pose(t: float, base_z: float) -> HeadPose:
    """Smooth head sway + nod, as a HeadPose the real encoder can consume."""
    x = 18.0 * np.sin(2 * np.pi * 0.15 * t)
    y = 9.0 * np.sin(2 * np.pi * 0.11 * t + 0.7)
    z = base_z + 8.0 * np.sin(2 * np.pi * 0.08 * t + 1.3)
    pitch = np.deg2rad(7.0 * np.sin(2 * np.pi * 0.13 * t))
    yaw = np.deg2rad(10.0 * np.sin(2 * np.pi * 0.10 * t + 0.4))
    roll = np.deg2rad(3.0 * np.sin(2 * np.pi * 0.17 * t + 1.1))
    rvec = np.array([pitch, yaw, roll], dtype=np.float64)
    tvec = np.array([x, y, z], dtype=np.float64)
    return HeadPose(
        rotation_vector=rvec.reshape(3, 1),
        translation_vector_mm=tvec,
        euler_degrees=(float(np.rad2deg(pitch)), float(np.rad2deg(yaw)), float(np.rad2deg(roll))),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.full(6, z, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=None,
        confidence=1.0,
        smoothed=True,
    )


def run_synthetic(cfg, duration, rate_hz, baseline_sec, out) -> int:
    encoder = CncOffsetEncoder(cfg)
    base_z = 600.0
    encoder.capture_baseline(_synthetic_pose(0.0, base_z), machine_pose=cfg.machine_pose)
    dt = 1.0 / rate_hz
    rows = []
    n = int(duration * rate_hz)
    for i in range(n):
        t = i * dt
        pose = _synthetic_pose(t, base_z)
        off = encoder.encode(pose, machine_pose=cfg.machine_pose)
        delta = encoder.head_delta_machine_mm(pose)
        rows.append(_row(t, pose.translation_vector_mm, pose.euler_degrees, delta, off, 1.0, True))
    _write(out, rows)
    print(f"synthetic: wrote {len(rows)} rows to {out}")
    return 0


def run_live(cfg, duration, rate_hz, baseline_sec, out) -> int:
    from orbbec_head_tracking.tracker import OrbbecHeadTracker

    encoder = CncOffsetEncoder(cfg)
    period = 1.0 / rate_hz
    rows = []
    base_t: list[np.ndarray] = []
    base_r: list[np.ndarray] = []
    with OrbbecHeadTracker() as tracker:
        start = time.monotonic()
        baseline_deadline = start + baseline_sec
        next_tick = start
        while True:
            now = time.monotonic()
            if now - start > duration:
                break
            if now < next_tick:
                time.sleep(min(0.001, next_tick - now))
                continue
            next_tick += period
            frame = tracker.read_frame()
            ok = frame is not None and frame.pose is not None
            if not ok:
                continue
            pose = frame.pose
            conf = float(getattr(pose, "confidence", 1.0))
            if not encoder.baseline_ready:
                if now <= baseline_deadline:
                    base_t.append(np.asarray(pose.translation_vector_mm, np.float64).reshape(3))
                    base_r.append(np.asarray(pose.rotation_vector, np.float64).reshape(3))
                    continue
                t_mean = np.mean(np.stack(base_t), axis=0) if base_t else pose.translation_vector_mm
                r_mean = np.mean(np.stack(base_r), axis=0) if base_r else pose.rotation_vector
                from dataclasses import replace
                averaged = replace(
                    pose,
                    translation_vector_mm=np.asarray(t_mean, np.float32),
                    rotation_vector=np.asarray(r_mean, np.float64).reshape(3, 1),
                )
                encoder.capture_baseline(averaged, machine_pose=cfg.machine_pose)
            off = encoder.encode(pose, machine_pose=cfg.machine_pose)
            delta = encoder.head_delta_machine_mm(pose)
            rows.append(_row(now - start, pose.translation_vector_mm, pose.pitch_yaw_roll,
                             delta, off, conf, True))
    _write(out, rows)
    print(f"live: wrote {len(rows)} rows to {out}")
    return 0


def _write(out: Path, rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("live", "synthetic"), default="synthetic")
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--duration", type=float, default=12.0, help="seconds")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--capture-baseline-sec", type=float, default=2.0)
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUT))
    args = parser.parse_args()

    cfg = _load_config(args.calibration)
    out = Path(args.output)
    if args.source == "synthetic":
        return run_synthetic(cfg, args.duration, args.rate_hz, args.capture_baseline_sec, out)
    return run_live(cfg, args.duration, args.rate_hz, args.capture_baseline_sec, out)


if __name__ == "__main__":
    raise SystemExit(main())
