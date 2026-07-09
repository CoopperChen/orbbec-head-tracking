#!/usr/bin/env python3
"""Show pose solver: depth-embedded rigid fit with anchor depths and head axes."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp

from capture import PipelineCapture, colorize_depth_mm
from orbbec_head_tracking.tracker import TrackerConfig, TrackingFrame
from pose_stage import PoseEstimator, render_pose_panel

_mp_face_mesh = mp.solutions.face_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline demo: pose solver stage")
    parser.add_argument(
        "--pose-solver",
        choices=("depth-rigid", "pnp", "hybrid"),
        default="depth-rigid",
        help="Pose estimation backend (default: depth-rigid)",
    )
    args = parser.parse_args()

    config = TrackerConfig(pose_solver=args.pose_solver, smoothing_enabled=False)
    window = f"Pipeline Demo: Pose solver ({args.pose_solver})"
    depth_window = f"{window} - aligned depth"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window, cv2.WINDOW_NORMAL)

    face_mesh = _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    try:
        with PipelineCapture() as capture:
            if capture.camera_matrix is None or capture.distortion_coefficients is None:
                raise RuntimeError("Camera calibration unavailable")
            pose_estimator = PoseEstimator(
                capture.camera_matrix,
                capture.distortion_coefficients,
                config,
            )
            while True:
                snapshot = capture.read()
                if snapshot is None:
                    continue
                rgb = snapshot.color_bgr
                result = face_mesh.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                if result.multi_face_landmarks:
                    frame = pose_estimator.estimate(
                        rgb,
                        snapshot.depth_aligned_mm,
                        result.multi_face_landmarks[0],
                    )
                    canvas = render_pose_panel(
                        frame,
                        capture.camera_matrix,
                        capture.distortion_coefficients,
                    )
                else:
                    frame = TrackingFrame(
                        color_bgr=rgb,
                        depth_mm=snapshot.depth_aligned_mm,
                        pose=None,
                    )
                    canvas = render_pose_panel(
                        frame,
                        capture.camera_matrix,
                        capture.distortion_coefficients,
                    )
                cv2.imshow(window, canvas)
                cv2.imshow(depth_window, colorize_depth_mm(snapshot.depth_aligned_mm))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return
    finally:
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
