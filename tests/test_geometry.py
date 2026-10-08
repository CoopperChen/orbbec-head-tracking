from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.geometry import (
    DepthRangeMedian,
    backproject_mm,
    depth_inlier_mask,
    median_positive_depth_mm,
    rotation_angle_deg,
    rotation_matrix_to_euler_degrees,
    slerp_rotation_matrices,
    stabilize_rotation_matrix,
    stabilize_translation_mm,
    translation_on_nose_ray_mm,
)
from orbbec_head_tracking.smoothing import PoseSmoother
from orbbec_head_tracking.types import HeadPose


def test_depth_inlier_mask_rejects_outlier() -> None:
    depths = np.array([500.0, 502.0, 498.0, 560.0], dtype=np.float64)
    keep = depth_inlier_mask(depths, max_deviation_mm=20.0)
    assert keep.tolist() == [True, True, True, False]


def test_stabilize_translation_mm_clamps_jump() -> None:
    prev = np.array([10.0, 0.0, 500.0], dtype=np.float32)
    jumped = np.array([40.0, 0.0, 500.0], dtype=np.float32)
    stabilized = stabilize_translation_mm(jumped, prev, max_jump_mm=10.0)
    assert stabilized[0] == pytest.approx(20.0)


def test_pose_smoother_norm_deadband_suppresses_small_motion() -> None:
    smoother = PoseSmoother(0.5, 0.5, 0.0, 0.5, translation_norm_deadband_mm=6.0)
    base = HeadPose(
        rotation_vector=np.zeros((3, 1), dtype=np.float32),
        translation_vector_mm=np.array([0.0, 0.0, 500.0], dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
    )
    smoother.smooth(base)
    jittered = HeadPose(
        rotation_vector=np.zeros((3, 1), dtype=np.float32),
        translation_vector_mm=np.array([2.0, -1.5, 501.0], dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
    )
    smoothed = smoother.smooth(jittered)
    assert float(np.linalg.norm(smoothed.translation_vector_mm - np.array([0.0, 0.0, 500.0]))) < 1.5


def test_rotation_matrix_to_euler_degrees_identity() -> None:
    rmat = np.eye(3, dtype=np.float32)
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(rmat)
    assert abs(pitch) < 1e-5
    assert abs(yaw) < 1e-5
    assert abs(roll) < 1e-5


def test_stabilize_rotation_matrix_rejects_jump() -> None:
    import cv2

    r0 = np.eye(3, dtype=np.float64)
    r1, _ = cv2.Rodrigues(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    stabilized = stabilize_rotation_matrix(r1, r0, max_jump_deg=10.0)
    assert np.allclose(stabilized, r0)


def test_slerp_rotation_matrices_endpoints() -> None:
    import cv2

    r0 = np.eye(3, dtype=np.float64)
    r1, _ = cv2.Rodrigues(np.array([0.0, 0.0, 0.2], dtype=np.float64))
    mid = slerp_rotation_matrices(r0, r1, 0.0)
    assert np.allclose(mid, r0)
    end = slerp_rotation_matrices(r0, r1, 1.0)
    assert rotation_angle_deg(end, r1) < 0.5


def _test_camera_matrix() -> np.ndarray:
    return np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_backproject_mm_principal_point_and_off_axis() -> None:
    camera = _test_camera_matrix()
    on_axis = backproject_mm(np.array([640.0, 360.0]), 500.0, camera)
    assert on_axis.tolist() == pytest.approx([0.0, 0.0, 500.0])
    off_axis = backproject_mm(np.array([720.0, 360.0]), 500.0, camera)
    assert off_axis.tolist() == pytest.approx([50.0, 0.0, 500.0])


def test_depth_range_median_rejects_single_frame_spike() -> None:
    depth_filter = DepthRangeMedian(5)
    filtered = [depth_filter.update(value) for value in (500.0, 500.0, 500.0, 800.0, 500.0)]
    assert filtered[3] == pytest.approx(500.0)
    assert filtered[4] == pytest.approx(500.0)
    depth_filter.reset()
    assert depth_filter.update(420.0) == pytest.approx(420.0)


def test_translation_on_nose_ray_ignores_depth_spike() -> None:
    camera = _test_camera_matrix()
    points_2d = np.array(
        [
            [720.0, 360.0],
            [700.0, 500.0],
            [500.0, 300.0],
            [900.0, 300.0],
        ],
        dtype=np.float32,
    )
    depth_filter = DepthRangeMedian(5)
    for _ in range(3):
        translation_on_nose_ray_mm(points_2d, 500.0, camera, depth_filter)
    spiked = translation_on_nose_ray_mm(points_2d, 800.0, camera, depth_filter)
    assert spiked is not None
    assert spiked.tolist() == pytest.approx([50.0, 0.0, 500.0])
    held = translation_on_nose_ray_mm(points_2d, float("nan"), camera, depth_filter)
    assert held is not None
    assert held.tolist() == pytest.approx([50.0, 0.0, 500.0])
    assert median_positive_depth_mm(np.array([500.0, np.nan, 0.0, 510.0])) == pytest.approx(505.0)
