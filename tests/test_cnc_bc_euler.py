from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import (
    BcAxisSign,
    BcCameraRvecMapping,
    BcEulerMapping,
    CncCompensationConfig,
    OffsetDeadbandConfig,
)
from orbbec_head_tracking.cnc_kinematics import (
    bc_from_camera_rvec_delta,
    bc_from_euler_delta,
    camera_rvec_delta_degrees,
    euler_delta_machine_degrees,
    rvec_to_matrix,
    shortest_arc_delta_deg,
)
from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder, CncUserOffset
from orbbec_head_tracking.types import HeadPose


def _rvec_pose(
    t_mm: tuple[float, float, float],
    rvec: tuple[float, float, float],
) -> HeadPose:
    r = np.array(rvec, dtype=np.float32).reshape(3, 1)
    return HeadPose(
        rotation_vector=r,
        translation_vector_mm=np.array(t_mm, dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
        smoothed=False,
    )


def test_shortest_arc_delta_avoids_wrap_jump() -> None:
    assert shortest_arc_delta_deg(170.0, -170.0) == pytest.approx(-20.0)
    assert shortest_arc_delta_deg(-170.0, 170.0) == pytest.approx(20.0)


def test_camera_rvec_roll_maps_to_b_yaw_maps_to_c() -> None:
    config = CncCompensationConfig(
        bc_mode="camera_rvec",
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_camera_rvec_mapping=BcCameraRvecMapping(b_axis="z", c_axis="y"),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    baseline = _rvec_pose((0.0, 0.0, 500.0), (0.0, 0.0, 0.0))
    encoder.capture_baseline(baseline)

    roll_pose = _rvec_pose(
        (0.0, 0.0, 500.0),
        (0.0, 0.0, np.deg2rad(6.0)),
    )
    yaw_pose = _rvec_pose(
        (0.0, 0.0, 500.0),
        (0.0, np.deg2rad(6.0), 0.0),
    )
    roll_offset = encoder.encode(roll_pose)
    yaw_offset = encoder.encode(yaw_pose)

    assert abs(roll_offset.b) > abs(roll_offset.c)
    assert abs(yaw_offset.c) > abs(yaw_offset.b)
    assert roll_offset.b == pytest.approx(6.0, abs=0.5)
    assert yaw_offset.c == pytest.approx(6.0, abs=0.5)


def test_euler_machine_mapping_for_gemini_mount() -> None:
    rotation = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    config = CncCompensationConfig(
        camera_to_machine_rotation=rotation,
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_mode="euler_machine",
        bc_euler_mapping=BcEulerMapping(b_axis="pitch", c_axis="roll"),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_rvec_pose((0.0, 0.0, 500.0), (0.0, 0.0, 0.0)))

    roll_offset = encoder.encode(
        _rvec_pose((0.0, 0.0, 500.0), (0.0, 0.0, np.deg2rad(6.0)))
    )
    yaw_offset = encoder.encode(
        _rvec_pose((0.0, 0.0, 500.0), (0.0, np.deg2rad(6.0), 0.0))
    )

    assert abs(roll_offset.b) > abs(roll_offset.c)
    assert abs(yaw_offset.c) > abs(yaw_offset.b)
    assert roll_offset.b == pytest.approx(-6.0, abs=0.5)
    assert yaw_offset.c == pytest.approx(-6.0, abs=0.5)


def test_camera_rvec_roll_does_not_use_normal_only_path() -> None:
    config = CncCompensationConfig(
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_mode="normal",
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder_normal = CncOffsetEncoder(config)
    encoder_rvec = CncOffsetEncoder(
        CncCompensationConfig(
            bc_mode="camera_rvec",
            offset_deadband=OffsetDeadbandConfig(enabled=False),
            bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
        )
    )
    baseline = _rvec_pose((0.0, 0.0, 500.0), (0.0, 0.0, 0.0))
    encoder_normal.capture_baseline(baseline)
    encoder_rvec.capture_baseline(baseline)
    roll_pose = _rvec_pose(
        (0.0, 0.0, 500.0),
        (0.0, 0.0, np.deg2rad(8.0)),
    )
    normal_offset = encoder_normal.encode(roll_pose)
    rvec_offset = encoder_rvec.encode(roll_pose)
    assert abs(rvec_offset.b) > 1.0
    assert abs(normal_offset.b) < abs(rvec_offset.b) * 0.5


def test_yaw_progression_stays_continuous() -> None:
    config = CncCompensationConfig(
        bc_mode="camera_rvec",
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_rvec_pose((0.0, 0.0, 500.0), (0.0, 0.0, 0.0)))
    last_c = 0.0
    for deg in range(1, 12):
        pose = _rvec_pose(
            (0.0, 0.0, 500.0),
            (0.0, np.deg2rad(float(deg)), 0.0),
        )
        offset = encoder.encode(pose)
        assert abs(offset.c - last_c) < 8.0
        last_c = offset.c


def test_camera_rvec_delta_matches_small_rotation() -> None:
    baseline = rvec_to_matrix(np.zeros(3))
    current = rvec_to_matrix(np.deg2rad([0.0, 4.0, -3.0]))
    rx, ry, rz = camera_rvec_delta_degrees(current, baseline)
    assert ry == pytest.approx(4.0, abs=0.2)
    assert rz == pytest.approx(-3.0, abs=0.2)
    b, c = bc_from_camera_rvec_delta(
        rx,
        ry,
        rz,
        b_axis="z",
        c_axis="y",
        b_sign=1.0,
        c_sign=1.0,
        follow_sign=1.0,
    )
    assert b == pytest.approx(-3.0, abs=0.2)
    assert c == pytest.approx(4.0, abs=0.2)


def test_euler_delta_uses_rotation_delta_matrix() -> None:
    rotation = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    baseline = rvec_to_matrix(np.deg2rad([2.0, -1.0, 0.5]))
    current = rvec_to_matrix(np.deg2rad([2.0, 5.0, 0.5]))
    pitch_d, yaw_d, roll_d = euler_delta_machine_degrees(current, baseline, rotation)
    b, c = bc_from_euler_delta(
        pitch_d,
        yaw_d,
        roll_d,
        b_axis="pitch",
        c_axis="roll",
        b_sign=1.0,
        c_sign=1.0,
        follow_sign=1.0,
    )
    assert abs(b) < 0.5
    assert c == pytest.approx(-6.0, abs=0.5)
