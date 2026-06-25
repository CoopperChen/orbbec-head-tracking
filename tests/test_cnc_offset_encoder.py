from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, MachinePose
from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder, CncUserOffset
from orbbec_head_tracking.cnc_safety import CncSafetyGuards
from orbbec_head_tracking.types import HeadPose


def _make_pose(
    t_mm: tuple[float, float, float],
    rvec: tuple[float, float, float] = (0.0, 0.0, 0.0),
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


def test_zero_delta_after_baseline() -> None:
    config = CncCompensationConfig(offset_mode="follow")
    encoder = CncOffsetEncoder(config)
    pose = _make_pose((100.0, 200.0, 800.0))
    encoder.capture_baseline(pose)
    offset = encoder.encode(pose)
    assert offset.x == pytest.approx(0.0, abs=1e-4)
    assert offset.y == pytest.approx(0.0, abs=1e-4)
    assert offset.z == pytest.approx(0.0, abs=1e-4)


def test_follow_positive_x_delta() -> None:
    config = CncCompensationConfig(offset_mode="follow")
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_make_pose((0.0, 0.0, 500.0)))
    moved = _make_pose((5.0, 0.0, 500.0))
    offset = encoder.encode(moved)
    assert offset.x == pytest.approx(5.0, abs=1e-4)
    assert offset.y == pytest.approx(0.0, abs=1e-4)


def test_counter_mode_negates_translation() -> None:
    config = CncCompensationConfig(offset_mode="counter")
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_make_pose((0.0, 0.0, 500.0)))
    moved = _make_pose((5.0, 0.0, 500.0))
    offset = encoder.encode(moved)
    assert offset.x == pytest.approx(-5.0, abs=1e-4)


def test_axis_limits_clamp() -> None:
    from orbbec_head_tracking.cnc_config import AxisLimits

    config = CncCompensationConfig(
        offset_mode="follow",
        axis_limits=AxisLimits(x_mm=(-2.0, 2.0)),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_make_pose((0.0, 0.0, 500.0)))
    moved = _make_pose((10.0, 0.0, 500.0))
    offset = encoder.encode(moved)
    assert offset.x == pytest.approx(2.0, abs=1e-4)


def test_safety_zeros_on_tracking_loss() -> None:
    config = CncCompensationConfig()
    safety = CncSafetyGuards(config)
    proposed = CncUserOffset(1.0, 2.0, 3.0, 4.0, 5.0)
    decision = safety.evaluate(
        proposed,
        tracking_ok=False,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "zero"
    assert decision.offset == CncUserOffset.zero()


def test_safety_low_confidence_zeros() -> None:
    config = CncCompensationConfig()
    safety = CncSafetyGuards(config)
    decision = safety.evaluate(
        CncUserOffset(1, 1, 1, 0, 0),
        tracking_ok=True,
        confidence=0.1,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "zero"
    assert decision.offset == CncUserOffset.zero()


def test_kinematic_solver_with_machine_pose() -> None:
    config = CncCompensationConfig(
        solver="kinematic",
        offset_mode="follow",
        machine_pose=MachinePose(-58.5, 32.38, 27.0, 77.21, 33.4),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_make_pose((0.0, 0.0, 600.0)))
    offset = encoder.encode(
        _make_pose((1.0, 0.5, 600.0)),
        machine_pose=config.machine_pose,
    )
    assert abs(offset.x) + abs(offset.y) + abs(offset.z) > 0.0


def test_spike_allows_rate_limited_ramp() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    target = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    last = CncUserOffset.zero()
    for _ in range(300):
        last = safety.evaluate(
            target,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    assert last.x == pytest.approx(5.0, abs=0.05)
    assert last.y == pytest.approx(0.0, abs=1e-4)


def test_spike_rejects_single_frame_jump() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(5):
        safety.evaluate(
            steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    decision = safety.evaluate(
        CncUserOffset(10.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "spike_rejected"
