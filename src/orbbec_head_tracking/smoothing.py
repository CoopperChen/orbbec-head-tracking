from __future__ import annotations

import cv2
import numpy as np

from .geometry import rotation_matrix_to_euler_degrees
from .types import HeadPose


class PoseSmoother:
    def __init__(self, translation_alpha: float, rotation_alpha: float, translation_deadband_mm: float, rotation_deadband_deg: float) -> None:
        self.translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))
        self.rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
        self.translation_deadband_mm = max(0.0, float(translation_deadband_mm))
        self.rotation_deadband_rad = float(np.radians(max(0.0, float(rotation_deadband_deg))))
        self.translation_vector_mm: np.ndarray | None = None
        self.rotation_vector: np.ndarray | None = None

    def reset(self) -> None:
        self.translation_vector_mm = None
        self.rotation_vector = None

    def smooth(self, pose: HeadPose) -> HeadPose:
        t = pose.translation_vector_mm.astype(np.float32).reshape(3)
        r = pose.rotation_vector.astype(np.float32).reshape(3)
        if self.translation_vector_mm is None or self.rotation_vector is None:
            self.translation_vector_mm = t.copy()
            self.rotation_vector = r.copy()
            return pose
        dt = t - self.translation_vector_mm
        dt = np.where(np.abs(dt) < self.translation_deadband_mm, 0.0, dt)
        self.translation_vector_mm = (self.translation_vector_mm + self.translation_alpha * dt).astype(np.float32)
        dr = r - self.rotation_vector
        dr = np.where(np.abs(dr) < self.rotation_deadband_rad, 0.0, dr)
        self.rotation_vector = (self.rotation_vector + self.rotation_alpha * dr).astype(np.float32)
        rmat, _ = cv2.Rodrigues(self.rotation_vector.reshape(3, 1))
        return HeadPose(
            rotation_vector=self.rotation_vector.reshape(3, 1).copy(),
            translation_vector_mm=self.translation_vector_mm.copy(),
            euler_degrees=rotation_matrix_to_euler_degrees(rmat),
            landmarks_2d=pose.landmarks_2d,
            sampled_depth_mm=pose.sampled_depth_mm,
            inliers=pose.inliers,
            solver=pose.solver,
            valid_depth_count=pose.valid_depth_count,
            reprojection_error_px=pose.reprojection_error_px,
            confidence=pose.confidence,
            smoothed=True,
        )
