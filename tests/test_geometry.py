from __future__ import annotations

import numpy as np

from orbbec_head_tracking.geometry import rotation_matrix_to_euler_degrees


def test_rotation_matrix_to_euler_degrees_identity() -> None:
    rmat = np.eye(3, dtype=np.float32)
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(rmat)
    assert abs(pitch) < 1e-5
    assert abs(yaw) < 1e-5
    assert abs(roll) < 1e-5
