from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PoseSolver


@dataclass(frozen=True)
class HeadPose:
    rotation_vector: np.ndarray
    translation_vector_mm: np.ndarray
    euler_degrees: tuple[float, float, float]
    landmarks_2d: np.ndarray
    sampled_depth_mm: np.ndarray
    inliers: np.ndarray | None
    solver: PoseSolver
    valid_depth_count: int
    reprojection_error_px: float | None
    confidence: float
    smoothed: bool = False

    @property
    def pitch_yaw_roll(self) -> tuple[float, float, float]:
        return self.euler_degrees


@dataclass(frozen=True)
class TrackingFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    pose: HeadPose | None
    frame_index: int | None = None
