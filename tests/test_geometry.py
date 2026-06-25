from __future__ import annotations

import numpy as np

from orbbec_head_tracking.geometry import (
    rotation_angle_deg,
    rotation_matrix_to_euler_degrees,
    slerp_rotation_matrices,
    stabilize_rotation_matrix,
)


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
