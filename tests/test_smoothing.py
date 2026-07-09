from __future__ import annotations

import numpy as np

from orbbec_head_tracking.smoothing import PoseSmoother, smooth_translation_step
from orbbec_head_tracking.types import HeadPose


def test_pose_smoother_reset() -> None:
    smoother = PoseSmoother(0.2, 0.2, 1.0, 1.0)
    smoother.reset()
    assert smoother.translation_vector_mm is None
    assert smoother.rotation_matrix is None


def test_pose_smoother_norm_deadband() -> None:
    smoother = PoseSmoother(0.4, 0.4, 0.0, 1.0, translation_norm_deadband_mm=5.0)
    pose = HeadPose(
        rotation_vector=np.zeros((3, 1), dtype=np.float32),
        translation_vector_mm=np.array([0.0, 0.0, 600.0], dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
    )
    smoother.smooth(pose)
    nudged = HeadPose(
        rotation_vector=np.zeros((3, 1), dtype=np.float32),
        translation_vector_mm=np.array([1.0, 1.0, 601.0], dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
    )
    out = smoother.smooth(nudged)
    assert float(np.linalg.norm(out.translation_vector_mm - np.array([0.0, 0.0, 600.0]))) < 0.8


def test_smooth_translation_step_follows_macro_motion() -> None:
    previous = np.array([0.0, 0.0, 600.0], dtype=np.float32)
    current = np.array([8.0, 0.0, 600.0], dtype=np.float32)
    stepped = smooth_translation_step(
        previous,
        current,
        alpha=0.2,
        per_axis_deadband_mm=0.0,
        norm_deadband_mm=2.5,
    )
    assert stepped[0] > 1.0
