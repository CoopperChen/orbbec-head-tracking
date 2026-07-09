#!/usr/bin/env python3
"""All pipeline stages in one 2x3 grid (single camera session)."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np

from capture import PipelineCapture, blend_rgb_depth, colorize_depth_mm
from orbbec_head_tracking.tracker import TrackerConfig, TrackingFrame
from pose_stage import PoseEstimator, render_pose_panel

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles

PANEL_WIDTH = 480
PANEL_HEIGHT = 360


def _fit_panel(image: np.ndarray, width: int = PANEL_WIDTH, height: int = PANEL_HEIGHT) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, title: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (16, 20, 28), -1)
    cv2.addWeighted(panel, 0.72, image, 0.28, 0.0, panel)
    cv2.putText(
        panel,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 245, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return panel


def _landmarks_panel(rgb: np.ndarray, face_mesh: Any) -> np.ndarray:
    canvas = rgb.copy()
    result = face_mesh.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            _mp_drawing.draw_landmarks(
                canvas,
                face_landmarks,
                _mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_styles.get_default_face_mesh_tesselation_style(),
            )
            _mp_drawing.draw_landmarks(
                canvas,
                face_landmarks,
                _mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_styles.get_default_face_mesh_contours_style(),
            )
    return canvas, result


def _no_pose_panel(rgb: np.ndarray) -> np.ndarray:
    canvas = rgb.copy()
    cv2.putText(
        canvas,
        "Return no pose frame",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline demo: all stages at once")
    parser.add_argument(
        "--pose-solver",
        choices=("depth-rigid", "pnp", "hybrid"),
        default="depth-rigid",
    )
    args = parser.parse_args()

    face_mesh = _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    window = "Pipeline Demo: All stages"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        with PipelineCapture() as capture:
            if capture.camera_matrix is None or capture.distortion_coefficients is None:
                raise RuntimeError("Camera calibration unavailable")
            pose_estimator = PoseEstimator(
                capture.camera_matrix,
                capture.distortion_coefficients,
                TrackerConfig(pose_solver=args.pose_solver, smoothing_enabled=False),
            )
            while True:
                snapshot = capture.read()
                if snapshot is None:
                    continue

                rgb = snapshot.color_bgr
                landmarks_canvas, mp_result = _landmarks_panel(rgb, face_mesh)

                if mp_result.multi_face_landmarks:
                    tracking_frame = pose_estimator.estimate(
                        rgb,
                        snapshot.depth_aligned_mm,
                        mp_result.multi_face_landmarks[0],
                    )
                    pose_canvas = render_pose_panel(
                        tracking_frame,
                        capture.camera_matrix,
                        capture.distortion_coefficients,
                    )
                else:
                    tracking_frame = TrackingFrame(
                        color_bgr=rgb,
                        depth_mm=snapshot.depth_aligned_mm,
                        pose=None,
                    )
                    pose_canvas = _no_pose_panel(rgb)

                panels = [
                    _label(_fit_panel(rgb), "RGB frames"),
                    _label(_fit_panel(snapshot.depth_native_vis), "Depth frames"),
                    _label(_fit_panel(landmarks_canvas), "Face landmarks (MediaPipe)"),
                    _label(
                        _fit_panel(blend_rgb_depth(rgb, snapshot.depth_aligned_mm)),
                        "Align depth with RGB",
                    ),
                    _label(
                        _fit_panel(colorize_depth_mm(snapshot.depth_aligned_mm)),
                        "Aligned depth (RGB resolution)",
                    ),
                    _label(_fit_panel(pose_canvas), f"Pose solver ({args.pose_solver})"),
                ]

                row_top = np.hstack(panels[:3])
                row_bottom = np.hstack(panels[3:])
                grid = np.vstack([row_top, row_bottom])

                cv2.imshow(window, grid)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return
    finally:
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
