"""Pose estimation for pipeline demos (single capture session)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from orbbec_head_tracking.tracker import (
    FACE_3D_MODEL,
    HeadPose,
    PoseSmoother,
    TrackerConfig,
    TrackingFrame,
    _fit_rigid_transform,
    _landmarks_to_points,
    _points_to_camera_3d,
    _rotation_matrix_to_euler_degrees,
    _sample_depth,
    draw_pose_overlay,
)


class PoseEstimator:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        config: TrackerConfig | None = None,
    ) -> None:
        self.config = config or TrackerConfig(smoothing_enabled=False)
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.pose_smoother = PoseSmoother(
            self.config.translation_alpha,
            self.config.rotation_alpha,
            self.config.translation_deadband_mm,
            self.config.rotation_deadband_deg,
        )
        self.missed_pose_count = 0
        self.previous_raw_rotation_vector: np.ndarray | None = None
        self.previous_raw_translation_vector_mm: np.ndarray | None = None

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        face_landmarks: Any,
    ) -> TrackingFrame:
        image_height, image_width = color_bgr.shape[:2]
        points_2d = _landmarks_to_points(face_landmarks, image_width, image_height)

        if self.config.pose_solver == "depth-rigid":
            object_points, camera_points, sampled_depth = _points_to_camera_3d(
                points_2d,
                depth_mm,
                self.camera_matrix,
                self.config.depth_sample_radius_px,
            )
            if len(camera_points) < self.config.min_depth_points:
                self._mark_pose_missed()
                return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=None)
            rvec, tvec = _fit_rigid_transform(object_points, camera_points)
            inliers = np.arange(len(camera_points), dtype=np.int32).reshape(-1, 1)
        else:
            sampled_depth = _sample_depth(depth_mm, points_2d)
            rvec_guess = self.previous_raw_rotation_vector
            tvec_guess = self.previous_raw_translation_vector_mm
            use_guess = (
                self.config.use_previous_pose_guess
                and rvec_guess is not None
                and tvec_guess is not None
            )
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                FACE_3D_MODEL,
                points_2d,
                self.camera_matrix,
                self.distortion_coefficients,
                rvec=rvec_guess.copy() if use_guess else None,
                tvec=tvec_guess.reshape(3, 1).copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                iterationsCount=self.config.pnp_iterations_count,
                reprojectionError=self.config.pnp_reprojection_error,
                confidence=self.config.pnp_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                self._mark_pose_missed()
                return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=None)

            if self.config.refine_pnp and inliers is not None and len(inliers) >= 6:
                inlier_indices = inliers.reshape(-1)
                rvec, tvec = cv2.solvePnPRefineLM(
                    FACE_3D_MODEL[inlier_indices],
                    points_2d[inlier_indices],
                    self.camera_matrix,
                    self.distortion_coefficients,
                    rvec,
                    tvec,
                )

        rmat, _ = cv2.Rodrigues(rvec)
        euler_degrees = _rotation_matrix_to_euler_degrees(rmat)
        self.previous_raw_rotation_vector = rvec.astype(np.float32)
        self.previous_raw_translation_vector_mm = tvec.reshape(3).astype(np.float32)
        pose = HeadPose(
            rotation_vector=self.previous_raw_rotation_vector,
            translation_vector_mm=self.previous_raw_translation_vector_mm,
            euler_degrees=euler_degrees,
            landmarks_2d=points_2d,
            sampled_depth_mm=sampled_depth,
            inliers=inliers,
        )
        self.missed_pose_count = 0
        if self.config.smoothing_enabled:
            pose = self.pose_smoother.smooth(pose)
        return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=pose)

    def _mark_pose_missed(self) -> None:
        self.missed_pose_count += 1
        if self.missed_pose_count >= self.config.reset_after_missed_frames:
            self.pose_smoother.reset()
            self.previous_raw_rotation_vector = None
            self.previous_raw_translation_vector_mm = None


def annotate_solver_depths(canvas: np.ndarray, frame: TrackingFrame) -> None:
    pose = frame.pose
    if pose is None:
        return
    for point, depth_mm in zip(pose.landmarks_2d, pose.sampled_depth_mm):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if np.isfinite(depth_mm) and depth_mm > 0.0:
            label = f"{depth_mm:.0f} mm"
            color = (0, 255, 255)
        else:
            label = "no depth"
            color = (80, 80, 255)
        cv2.putText(
            canvas,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            lineType=cv2.LINE_AA,
        )


def render_pose_panel(frame: TrackingFrame, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    canvas = draw_pose_overlay(frame, camera_matrix, distortion_coefficients)
    annotate_solver_depths(canvas, frame)
    return canvas
