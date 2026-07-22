"""Capture one tracked frame for fig4_tracking.py --frames.

Interactive preview (default):
  space  freeze / unfreeze candidate frame
  s      save current (or frozen) frame to .npz
  q/Esc  quit

  python scripts/figures/capture_fig4_frames.py
  python scripts/figures/capture_fig4_frames.py --auto   # first good frame, no UI

Then:
  python scripts/figures/fig4_tracking.py --frames results/fig4_frames.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from orbbec_head_tracking.tracker import OrbbecHeadTracker, colorize_depth_mm, draw_pose_overlay
from orbbec_head_tracking.types import TrackingFrame

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _REPO / "results" / "fig4_frames.npz"


@dataclass
class Candidate:
    frame: TrackingFrame
    rgb_vis: np.ndarray
    depth_vis: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Save the first tracked frame and exit (no preview)",
    )
    parser.add_argument("--window-name", type=str, default="Capture fig4 frames")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with OrbbecHeadTracker() as tracker:
        if args.auto:
            return _run_auto(tracker, out)
        return _run_interactive(tracker, out, args.window_name)


def _run_auto(tracker: OrbbecHeadTracker, out: Path) -> int:
    print("waiting for first tracked frame...")
    while True:
        frame = tracker.read_frame()
        if frame is None or frame.pose is None:
            continue
        _write(out, frame.color_bgr, frame.pose.landmarks_2d)
        print(f"wrote {out}")
        return 0


def _run_interactive(tracker: OrbbecHeadTracker, out: Path, window_name: str) -> int:
    if tracker.camera_matrix is None or tracker.distortion_coefficients is None:
        raise RuntimeError("Tracker calibration unavailable")

    rgb_win = window_name
    depth_win = f"{window_name} Depth"
    cv2.namedWindow(rgb_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_win, cv2.WINDOW_NORMAL)

    frozen: Candidate | None = None
    message = ""
    message_frames_left = 0
    print("keys: space=freeze/unfreeze  s=save  q=quit")

    try:
        while True:
            if frozen is None:
                frame = tracker.read_frame()
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        return 1
                    continue
                rgb_vis = draw_pose_overlay(
                    frame,
                    tracker.camera_matrix,
                    tracker.distortion_coefficients,
                )
                depth_vis = colorize_depth_mm(frame.depth_mm)
                live = Candidate(frame=frame, rgb_vis=rgb_vis, depth_vis=depth_vis)
                display = live
            else:
                display = frozen
                # Keep the pipeline moving so the camera does not stall while frozen.
                _ = tracker.read_frame()

            canvas = display.rgb_vis.copy()
            depth_canvas = display.depth_vis.copy()
            _draw_hud(canvas, display.frame, frozen is not None, message)
            cv2.imshow(rgb_win, canvas)
            cv2.imshow(depth_win, depth_canvas)

            if message_frames_left > 0:
                message_frames_left -= 1
                if message_frames_left == 0:
                    message = ""

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("quit without saving")
                return 1

            if key == ord(" "):
                if frozen is None:
                    if display.frame.pose is None:
                        message = "NO POSE - cannot freeze"
                        message_frames_left = 60
                        print(message)
                        continue
                    frozen = Candidate(
                        frame=display.frame,
                        rgb_vis=display.rgb_vis.copy(),
                        depth_vis=display.depth_vis.copy(),
                    )
                    message = "FROZEN - press s to save, space to resume"
                    message_frames_left = 90
                    print("frozen")
                else:
                    frozen = None
                    message = "LIVE"
                    message_frames_left = 45
                    print("live")
                continue

            if key in (ord("s"), ord("S")):
                candidate = frozen.frame if frozen is not None else display.frame
                if candidate.pose is None:
                    message = "NO POSE - wait for lock (or freeze a good frame)"
                    message_frames_left = 75
                    print(message)
                    continue
                _write(out, candidate.color_bgr, candidate.pose.landmarks_2d)
                message = f"SAVED {out.name}"
                message_frames_left = 90
                print(f"wrote {out}")
                # Brief flash so the user sees confirmation, then exit.
                _draw_hud(canvas, candidate, frozen is not None, message)
                cv2.imshow(rgb_win, canvas)
                cv2.waitKey(500)
                return 0
    finally:
        cv2.destroyAllWindows()


def _draw_hud(
    canvas: np.ndarray,
    frame: TrackingFrame,
    frozen: bool,
    message: str,
) -> None:
    h, w = canvas.shape[:2]
    pose = frame.pose
    tracking_ok = pose is not None
    conf = float(getattr(pose, "confidence", 0.0)) if pose is not None else 0.0

    # Top status bar.
    mode = "FROZEN" if frozen else "LIVE"
    mode_color = (0, 165, 255) if frozen else (80, 200, 80)
    track_text = "TRACKING OK" if tracking_ok else "NO TRACKING"
    track_color = (80, 200, 80) if tracking_ok else (40, 40, 220)
    conf_text = f"conf={conf:.2f}" if tracking_ok else "conf=--"

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    cv2.putText(
        canvas, mode, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, track_text, (150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, conf_text, (380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "space=freeze/resume   s=save   q=quit",
        (16, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if pose is not None:
        x_mm, y_mm, z_mm = [float(v) for v in pose.translation_vector_mm]
        pitch, yaw, roll = pose.pitch_yaw_roll
        pose_line = (
            f"xyz=({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) mm   "
            f"pyr=({pitch:.1f},{yaw:.1f},{roll:.1f}) deg"
        )
        cv2.putText(
            canvas,
            pose_line,
            (16, h - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    if message:
        color = (0, 220, 255) if "SAVED" in message or "FROZEN" in message or message == "LIVE" else (40, 40, 220)
        cv2.putText(
            canvas,
            message,
            (16, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def _write(out: Path, color_bgr: np.ndarray, landmarks_2d: np.ndarray) -> None:
    np.savez_compressed(
        out,
        color_bgr=np.asarray(color_bgr),
        landmarks_2d=np.asarray(landmarks_2d, dtype=np.float32),
    )


if __name__ == "__main__":
    raise SystemExit(main())
