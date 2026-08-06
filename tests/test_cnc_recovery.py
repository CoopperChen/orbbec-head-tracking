"""Post-hold recovery is bounded by wall-clock time and never hands off to catch_up."""

from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, SafetyConfig
from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.cnc_safety import CncSafetyGuards

TICK = 0.010
NORMAL_STEP_MM = 0.6


def _guards(**safety_overrides: object) -> CncSafetyGuards:
    config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(**safety_overrides),  # type: ignore[arg-type]
    )
    return CncSafetyGuards(config)


def _tick(guards: CncSafetyGuards, target: CncUserOffset, now_sec: float):
    return guards.evaluate(
        target,
        spike_reference=target,
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
        now_sec=now_sec,
    )


def test_minimum_recovery_is_a_duration_not_a_tick_count() -> None:
    """20 ticks was tuned at 10 ms; a 50 ms loop should not stretch it to a second."""
    guards = _guards()
    assert guards.recovery_sec_after_hold == pytest.approx(0.2)

    target = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    guards._state.last_offset = target
    guards.notify_hold("tracking_lost")

    now = 0.0
    ticks = 0
    while guards.in_recovery and ticks < 40:
        now += 0.050
        _tick(guards, target, now)
        ticks += 1

    assert not guards.in_recovery
    assert ticks <= 6


def test_recovery_never_hands_an_open_gap_to_catch_up() -> None:
    """The whole point of the ramp: no 3x burst the instant it expires."""
    guards = _guards()
    target = CncUserOffset(20.0, 0.0, 0.0, 0.0, 0.0)
    guards.notify_hold("tracking_lost")

    now = 0.0
    previous = 0.0
    largest_step = 0.0
    for _ in range(200):
        now += TICK
        decision = _tick(guards, target, now)
        largest_step = max(largest_step, abs(decision.offset.x - previous))
        previous = decision.offset.x

    assert previous == pytest.approx(20.0, abs=1e-6)
    assert largest_step <= NORMAL_STEP_MM + 1e-9


def test_recovery_outlives_its_minimum_while_the_gap_stays_open() -> None:
    guards = _guards()
    target = CncUserOffset(20.0, 0.0, 0.0, 0.0, 0.0)
    guards.notify_hold("tracking_lost")

    now = 0.0
    for _ in range(int(guards.recovery_sec_after_hold / TICK) + 2):
        now += TICK
        _tick(guards, target, now)

    assert guards.in_recovery
    assert guards.last_offset.x < 20.0


def test_recovery_ends_once_the_gap_closes() -> None:
    guards = _guards()
    target = CncUserOffset(0.4, 0.0, 0.0, 0.0, 0.0)
    guards.notify_hold("tracking_lost")

    now = 0.0
    for _ in range(int(guards.recovery_sec_after_hold / TICK) + 2):
        now += TICK
        _tick(guards, target, now)

    assert not guards.in_recovery
    assert guards.last_offset.x == pytest.approx(0.4)


def test_recovery_max_sec_releases_an_unreachable_target() -> None:
    guards = _guards(recovery_max_sec=0.05)
    target = CncUserOffset(200.0, 0.0, 0.0, 0.0, 0.0)
    guards.notify_hold("tracking_lost")

    now = 0.0
    for _ in range(20):
        now += TICK
        _tick(guards, target, now)

    assert not guards.in_recovery


def test_a_second_hold_rearms_recovery() -> None:
    guards = _guards()
    target = CncUserOffset(1.0, 0.0, 0.0, 0.0, 0.0)
    guards.notify_hold("tracking_lost")
    now = 0.0
    for _ in range(40):
        now += TICK
        _tick(guards, target, now)
    assert not guards.in_recovery

    guards.notify_hold("spike_rejected")
    assert guards.in_recovery
