from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, MachinePose, MismatchConfig
from orbbec_head_tracking.cnc_mismatch import CncMismatchTracker
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


def test_mismatch_disabled_passthrough() -> None:
    config = CncCompensationConfig(mismatch=MismatchConfig(enabled=False))
    tracker = CncMismatchTracker(config)
    required = CncUserOffset(3.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    target, report = tracker.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert target == required
    assert report.mode == "disabled"
    assert report.error_norm_mm == pytest.approx(2.0)


def test_mismatch_snap_disabled_by_default() -> None:
    tracker = CncMismatchTracker(CncCompensationConfig())
    required = CncUserOffset(4.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    _target, report = tracker.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert report.mode == "track"


def test_mismatch_suppresses_boost_after_command_hold() -> None:
    config = CncCompensationConfig(
        mismatch=MismatchConfig(enabled=True, kp=1.0, recovery_ticks_after_hold=5),
    )
    tracker = CncMismatchTracker(config)
    required = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    tracker.notify_command_hold("spike_rejected")
    target, report = tracker.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert report.mode == "track"
    assert target.x == pytest.approx(5.0)


def test_catch_up_rate_limit_converges_faster() -> None:
    from orbbec_head_tracking.cnc_config import SafetyConfig

    slow_config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(catch_up_multiplier=1.0),
    )
    fast_config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(catch_up_multiplier=3.0, catch_up_error_mm=0.5),
    )
    slow = CncSafetyGuards(slow_config)
    fast = CncSafetyGuards(fast_config)
    target = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    vision = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    slow_last = CncUserOffset.zero()
    fast_last = CncUserOffset.zero()
    for _ in range(8):
        slow_last = slow.evaluate(
            target,
            spike_reference=vision,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=5.0,
        ).offset
        fast_last = fast.evaluate(
            target,
            spike_reference=vision,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=5.0,
        ).offset
    assert fast_last.x > slow_last.x
    assert fast_last.x == pytest.approx(5.0, abs=0.2)


def test_mismatch_kp_boosts_target() -> None:
    config = CncCompensationConfig(
        mismatch=MismatchConfig(enabled=True, kp=1.0, ki=0.0, snap_error_mm=100.0),
    )
    tracker = CncMismatchTracker(config)
    required = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0)
    target, report = tracker.target(
        required,
        sent,
        dt_sec=0.01,
        head_speed_mm_s=20.0,
    )
    assert report.mode == "pi"
    assert target.x == pytest.approx(8.0)
    assert report.offset_error.x == pytest.approx(3.0)


def test_mismatch_snap_when_stable() -> None:
    config = CncCompensationConfig(
        mismatch=MismatchConfig(
            enabled=True,
            kp=0.0,
            snap_enabled=True,
            snap_head_speed_mm_s=5.0,
            snap_error_mm=1.0,
        ),
    )
    tracker = CncMismatchTracker(config)
    required = CncUserOffset(4.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    target, report = tracker.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert report.mode == "snap"
    assert target == required


def test_mismatch_snap_bypasses_rate_limit() -> None:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        mismatch=MismatchConfig(
            enabled=True,
            kp=0.0,
            snap_enabled=True,
            snap_head_speed_mm_s=5.0,
            snap_error_mm=1.0,
        ),
    )
    mismatch = CncMismatchTracker(config)
    safety = CncSafetyGuards(config)
    required = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    target, report = mismatch.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert report.mode == "snap"
    decision = safety.evaluate(
        target,
        spike_reference=required,
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
        snap=True,
    )
    assert decision.offset.x == pytest.approx(5.0)


def test_mismatch_ki_builds_correction_under_sustained_lag() -> None:
    config = CncCompensationConfig(
        mismatch=MismatchConfig(enabled=True, kp=0.0, ki=5.0, snap_error_mm=100.0),
    )
    tracker = CncMismatchTracker(config)
    required = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0)
    target_first, _ = tracker.target(required, sent, dt_sec=0.1, head_speed_mm_s=20.0)
    for _ in range(10):
        target_last, _ = tracker.target(required, sent, dt_sec=0.1, head_speed_mm_s=20.0)
    assert target_last.x > target_first.x
    assert target_last.x > required.x


def test_spike_reference_ignores_boosted_target() -> None:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        mismatch=MismatchConfig(enabled=True, kp=2.0, snap_error_mm=100.0),
    )
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(5):
        safety.evaluate(
            steady,
            spike_reference=steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )

    required = CncUserOffset(10.0, 0.0, 0.0, 0.0, 0.0)
    boosted = CncUserOffset(30.0, 0.0, 0.0, 0.0, 0.0)
    decision = safety.evaluate(
        boosted,
        spike_reference=required,
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "spike_rejected"


def test_spike_reference_allows_stable_boosted_ramp() -> None:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        mismatch=MismatchConfig(enabled=True, kp=1.0, snap_error_mm=100.0),
    )
    safety = CncSafetyGuards(config)
    required = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset.zero()
    for _ in range(300):
        error = required.x - sent.x
        target = CncUserOffset(required.x + error, 0.0, 0.0, 0.0, 0.0)
        sent = safety.evaluate(
            target,
            spike_reference=required,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=20.0,
        ).offset
    assert sent.x == pytest.approx(5.0, abs=0.5)


def test_mismatch_reset_on_baseline_capture() -> None:
    config = CncCompensationConfig(
        mismatch=MismatchConfig(enabled=True, ki=1.0),
    )
    tracker = CncMismatchTracker(config)
    tracker.capture_baseline(MachinePose(1.0, 2.0, 3.0, 4.0, 5.0))
    required = CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0)
    sent = CncUserOffset.zero()
    tracker.target(required, sent, dt_sec=0.1, head_speed_mm_s=20.0)
    assert tracker.last_report is not None
    tracker.capture_baseline(MachinePose(1.0, 2.0, 3.0, 4.0, 5.0))
    assert tracker.last_report is None
