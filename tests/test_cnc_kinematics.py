from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import (
    BcAxisSign,
    CncCompensationConfig,
    MachinePose,
    OffsetDeadbandConfig,
)
from orbbec_head_tracking.cnc_kinematics import (
    MachineConfig,
    bc_from_normal,
    nozzle_tip,
    solve_bc_delta_for_normal_target,
    solve_xyz_for_tip_delta,
    tool_normal_from_bc,
)


def test_nozzle_tip_at_machine_zero_pose() -> None:
    machine = MachineConfig()
    tip = nozzle_tip(0.0, 0.0, 0.0, 0.0, 90.0, machine)
    assert tip[0] == pytest.approx(0.0, abs=0.1)
    assert tip[1] == pytest.approx(-machine.a_mm, abs=0.1)
    assert tip[2] == pytest.approx(-machine.d_mm, abs=0.1)


def test_bc_from_normal_layout_design_fallback() -> None:
    from orbbec_head_tracking.cnc_kinematics import (
        bc_from_normal,
        bc_from_normal_layout_design,
        layout_design_available,
    )

    n = np.array([0.0, 0.0, 1.0])
    vendored = bc_from_normal(n)
    bridged = bc_from_normal_layout_design(n)
    if not layout_design_available():
        assert bridged == pytest.approx(vendored, abs=1e-6)


def test_solve_xyz_tip_delta_moves_tip() -> None:
    machine = MachineConfig()
    x0, y0, z0 = 0.0, 0.0, 0.0
    b0, c0 = 0.0, 90.0
    tip0 = nozzle_tip(x0, y0, z0, b0, c0, machine)
    x1, y1, z1 = solve_xyz_for_tip_delta(
        x0, y0, z0, b0, c0, np.array([1.0, 0.0, 0.0]), machine
    )
    tip1 = nozzle_tip(x1, y1, z1, b0, c0, machine)
    delta = tip1 - tip0
    assert np.linalg.norm(delta) == pytest.approx(1.0, abs=0.15)


def test_tool_normal_ik_zero_at_baseline() -> None:
    machine = MachineConfig()
    b, c = 12.0, 45.0
    n = tool_normal_from_bc(b, c, machine)
    assert solve_bc_delta_for_normal_target(n, b, c, machine) == pytest.approx((0.0, 0.0), abs=1e-3)


def test_tool_normal_ik_depends_on_machine_pose() -> None:
    from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder
    from orbbec_head_tracking.types import HeadPose

    def pose(rv: tuple[float, float, float]) -> HeadPose:
        r = np.array(rv, dtype=np.float32).reshape(3, 1)
        return HeadPose(
            rotation_vector=r,
            translation_vector_mm=np.array([0.0, 0.0, 600.0], dtype=np.float32),
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

    config = CncCompensationConfig(
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(
        pose((0.0, 0.0, 0.0)),
        machine_pose=MachinePose(0.0, 0.0, 0.0, 0.0, 90.0),
    )
    roll = pose((0.0, np.deg2rad(6.0), 0.0))
    low_c = encoder.encode(roll, machine_pose=MachinePose(0.0, 0.0, 0.0, 5.0, 30.0))
    high_c = encoder.encode(roll, machine_pose=MachinePose(0.0, 0.0, 0.0, -20.0, 120.0))
    assert abs(low_c.b) + abs(low_c.c) > 0.5
    assert abs(low_c.b - high_c.b) + abs(low_c.c - high_c.c) > 0.5


def test_camera_rvec_bc_is_pose_independent() -> None:
    from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder
    from orbbec_head_tracking.types import HeadPose

    def pose(rv: tuple[float, float, float]) -> HeadPose:
        r = np.array(rv, dtype=np.float32).reshape(3, 1)
        return HeadPose(
            rotation_vector=r,
            translation_vector_mm=np.array([0.0, 0.0, 600.0], dtype=np.float32),
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

    config = CncCompensationConfig(
        bc_mode="camera_rvec",
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(pose((0.0, 0.0, 0.0)))
    roll = pose((0.0, 0.0, np.deg2rad(6.0)))
    a = encoder.encode(roll, machine_pose=MachinePose(0.0, 0.0, 0.0, 0.0, 30.0))
    b = encoder.encode(roll, machine_pose=MachinePose(10.0, 20.0, 5.0, 12.0, 120.0))
    assert a.b == pytest.approx(b.b, abs=1e-4)
    assert a.c == pytest.approx(b.c, abs=1e-4)
