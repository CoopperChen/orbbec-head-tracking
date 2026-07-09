from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import (
    BcAxisSign,
    CncCompensationConfig,
    MachinePose,
    OffsetDeadbandConfig,
)
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


def test_safety_hold_last_on_tracking_loss_by_default() -> None:
    config = CncCompensationConfig()
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(1.0, 2.0, 3.0, 4.0, 5.0)
    for _ in range(200):
        safety.evaluate(
            steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    decision = safety.evaluate(
        CncUserOffset.zero(),
        tracking_ok=False,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "tracking_lost"
    assert decision.offset == steady


def test_safety_zeros_on_tracking_loss_when_configured() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        safety=SafetyConfig(on_tracking_loss="zero_offsets", on_low_confidence="zero_offsets"),
    )
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


def test_safety_hold_last_on_low_confidence_by_default() -> None:
    config = CncCompensationConfig()
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(1.0, 2.0, 3.0, 0.0, 0.0)
    for _ in range(200):
        safety.evaluate(
            steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    decision = safety.evaluate(
        CncUserOffset(9.0, 9.0, 9.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=0.1,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "low_confidence"
    assert decision.offset == steady


def test_safety_low_confidence_zeros_when_configured() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        safety=SafetyConfig(on_tracking_loss="zero_offsets", on_low_confidence="zero_offsets"),
    )
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


def test_safety_hold_last_on_head_speed_exceeded_by_default() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        safety=SafetyConfig(on_head_speed_exceeded="hold_last", head_speed_exceed_ticks=1),
    )
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(4.0, -1.0, 0.5, 0.2, -0.3)
    for _ in range(200):
        safety.evaluate(
            steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    decision = safety.evaluate(
        CncUserOffset(8.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=999.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "head_speed_exceeded"
    assert decision.offset == steady


def test_safety_head_speed_zeros_when_configured() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        safety=SafetyConfig(on_head_speed_exceeded="zero_offsets", head_speed_exceed_ticks=1),
    )
    safety = CncSafetyGuards(config)
    decision = safety.evaluate(
        CncUserOffset(8.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=999.0,
    )
    assert decision.action == "zero"
    assert decision.reason == "head_speed_exceeded"
    assert decision.offset == CncUserOffset.zero()


def test_safety_head_speed_exceed_requires_consecutive_ticks() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        safety=SafetyConfig(on_head_speed_exceeded="hold_last", head_speed_exceed_ticks=3),
    )
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(1.5, 0.0, 0.0, 0.0, 0.0)
    for _ in range(40):
        safety.evaluate(
            steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    for _ in range(2):
        decision = safety.evaluate(
            CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0),
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=999.0,
        )
        assert decision.action == "pass"
    decision = safety.evaluate(
        CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=999.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "head_speed_exceeded"


def test_bc_axis_sign_flips_rotation_offsets() -> None:
    import cv2

    config_pos = CncCompensationConfig(
        offset_mode="follow",
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    config_neg = CncCompensationConfig(
        offset_mode="follow",
        offset_deadband=OffsetDeadbandConfig(enabled=False),
        bc_axis_sign=BcAxisSign(b=-1.0, c=-1.0),
    )
    baseline = _make_pose((0.0, 0.0, 500.0))
    yaw_rvec = np.array([0.0, np.deg2rad(4.0), 0.0], dtype=np.float32).reshape(3, 1)
    moved = _make_pose((0.0, 0.0, 500.0), tuple(float(v) for v in yaw_rvec.reshape(3)))

    encoder_pos = CncOffsetEncoder(config_pos)
    encoder_neg = CncOffsetEncoder(config_neg)
    encoder_pos.capture_baseline(baseline)
    encoder_neg.capture_baseline(baseline)

    offset_pos = encoder_pos.encode(moved)
    offset_neg = encoder_neg.encode(moved)
    if abs(offset_pos.b) > 1e-4:
        assert offset_neg.b == pytest.approx(-offset_pos.b, rel=1e-3)
    if abs(offset_pos.c) > 1e-4:
        assert offset_neg.c == pytest.approx(-offset_pos.c, rel=1e-3)


def test_head_normal_uses_machine_frame_for_bc() -> None:
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
        bc_axis_sign=BcAxisSign(b=1.0, c=1.0),
    )
    encoder = CncOffsetEncoder(config)
    identity_r = np.eye(3, dtype=np.float64)
    n_machine = encoder._head_normal_machine(identity_r)
    n_expected = rotation @ np.array([0.0, 0.0, 1.0])
    n_expected = n_expected / np.linalg.norm(n_expected)
    assert np.allclose(n_machine, n_expected, atol=1e-6)


def test_kinematic_solver_with_machine_pose() -> None:
    from orbbec_head_tracking.cnc_config import OffsetDeadbandConfig

    config = CncCompensationConfig(
        solver="kinematic",
        offset_mode="follow",
        machine_pose=MachinePose(-58.5, 32.38, 27.0, 77.21, 33.4),
        offset_deadband=OffsetDeadbandConfig(enabled=False),
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


def test_spike_ignores_bc_axis_jitter() -> None:
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
        CncUserOffset(0.0, 0.0, 0.0, 8.0, 8.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "pass"


def test_spike_recovers_after_stable_return_to_baseline() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(
            vmax_mm_s=(30.0, 30.0, 15.0),
            vmax_deg_s=(10.0, 10.0),
        ),
    )
    safety = CncSafetyGuards(config)
    for _ in range(20):
        safety.evaluate(
            CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0),
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    safety.evaluate(
        CncUserOffset(10.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    decision = CncUserOffset.zero()
    for value in np.linspace(5.0, 0.0, 40):
        proposed = CncUserOffset(float(value), 0.0, 0.0, 0.0, 0.0)
        decision = safety.evaluate(
            proposed,
            spike_reference=proposed,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    for _ in range(120):
        decision = safety.evaluate(
            CncUserOffset(0.0, 0.0, 0.0, 0.0, 0.0),
            spike_reference=CncUserOffset(0.0, 0.0, 0.0, 0.0, 0.0),
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    assert decision.action == "pass"
    assert decision.reason in ("", "zero_settled", "zero_ramp")
    assert decision.offset.x == pytest.approx(0.0, abs=0.15)
