from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, SafetyConfig
from orbbec_head_tracking.cnc_mismatch import CncMismatchTracker
from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.cnc_safety import CncSafetyGuards


def test_tracking_loss_keeps_last_command_not_zero_target() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    held = CncUserOffset(3.0, 1.0, 0.5, 2.0, -4.0)
    for _ in range(20):
        held = safety.evaluate(
            held,
            spike_reference=held,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    decision = safety.evaluate(
        CncUserOffset.zero(),
        tracking_ok=False,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.action == "hold_last"
    assert decision.offset == held


def test_link_fault_holds_compensation_instead_of_zero() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(4.0, 0.0, 0.0, 1.0, 0.0)
    for _ in range(10):
        steady = safety.evaluate(
            steady,
            spike_reference=steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    decision = safety.evaluate(
        CncUserOffset.zero(),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
        link_ok=False,
    )
    assert decision.action == "hold_last"
    assert decision.reason == "udp_link_fault"
    assert decision.offset == steady


def test_recovery_ignores_zero_target_while_compensated() -> None:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(recovery_ticks_after_hold=5),
    )
    safety = CncSafetyGuards(config)
    steady = CncUserOffset(6.0, 0.0, 0.0, 0.0, 0.0)
    for _ in range(20):
        safety.evaluate(
            steady,
            spike_reference=steady,
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        )
    assert safety.last_offset.x == pytest.approx(6.0, abs=0.1)
    safety.evaluate(
        steady,
        tracking_ok=False,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert safety.in_recovery
    decision = safety.evaluate(
        CncUserOffset.zero(),
        spike_reference=CncUserOffset.zero(),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.offset.x == pytest.approx(6.0, abs=0.1)
    assert decision.reason in ("recovery_hold", "recovery_ramp", "catch_up", "")


def test_mismatch_preserves_sent_during_recovery() -> None:
    config = CncCompensationConfig()
    tracker = CncMismatchTracker(config)
    sent = CncUserOffset(5.0, 0.0, 0.0, 3.0, 0.0)
    target, report = tracker.target(
        CncUserOffset.zero(),
        sent,
        dt_sec=0.01,
        head_speed_mm_s=0.0,
        preserve_sent=True,
    )
    assert target == sent
    assert report.mode == "preserve_sent"
