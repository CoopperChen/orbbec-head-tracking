from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, MismatchConfig
from orbbec_head_tracking.cnc_mismatch import CncMismatchTracker
from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.cnc_safety import CncSafetyGuards


def test_spike_hold_updates_last_proposed_to_rejected_vision() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(2.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(50):
        safety.evaluate(
            steady,
            spike_reference=steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    safety.evaluate(
        CncUserOffset(10.0, 0.0, 0.0, 0.0, 0.0),
        spike_reference=CncUserOffset(10.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert safety._state.last_proposed.x == pytest.approx(10.0)


def test_hold_recovery_tracks_required_without_overshoot() -> None:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        mismatch=MismatchConfig(enabled=True, kp=1.0, recovery_ticks_after_hold=5),
    )
    mismatch = CncMismatchTracker(config)
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(3.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(80):
        safety.evaluate(
            steady,
            spike_reference=steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    hold = safety.evaluate(
        CncUserOffset(12.0, 0.0, 0.0, 0.0, 0.0),
        spike_reference=CncUserOffset(12.0, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert hold.action == "hold_last"
    mismatch.notify_command_hold(hold.reason)
    required = CncUserOffset(3.0, 0.0, 0.0, 0.0, 0.0)
    last = safety.last_offset
    for _ in range(120):
        target, _ = mismatch.target(
            required,
            safety.last_offset,
            dt_sec=0.01,
            head_speed_mm_s=0.0,
        )
        last = safety.evaluate(
            target,
            spike_reference=required,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    assert last.x == pytest.approx(3.0, abs=0.2)


def test_tracking_loss_recovery_does_not_jump_to_limit() -> None:
    from orbbec_head_tracking.cnc_config import AxisLimits, SafetyConfig

    config = CncCompensationConfig(
        update_period_ms=10.0,
        axis_limits=AxisLimits(c_deg=(-15.0, 15.0)),
        safety=SafetyConfig(
            vmax_deg_s=(25.0, 25.0),
            catch_up_multiplier=3.0,
            recovery_ticks_after_hold=20,
        ),
    )
    safety = CncSafetyGuards(config)
    held = CncUserOffset(0.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(30):
        held = safety.evaluate(
            held,
            spike_reference=held,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    safety.evaluate(
        held,
        tracking_ok=False,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    saturated = CncUserOffset(0.0, 0.0, 0.0, 0.0, 15.0)
    last = held
    for _ in range(5):
        last = safety.evaluate(
            saturated,
            spike_reference=saturated,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    assert abs(last.c) < 2.0
    assert abs(last.c) < abs(saturated.c)
