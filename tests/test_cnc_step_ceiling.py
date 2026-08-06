"""The rate limiter honours both a velocity limit and a single-packet step ceiling."""

from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, SafetyConfig
from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.cnc_safety import MAX_PERIOD_SEC, CncSafetyGuards


def _guards(**safety_overrides: object) -> CncSafetyGuards:
    overrides: dict[str, object] = {"catch_up_multiplier": 1.0}
    overrides.update(safety_overrides)
    config = CncCompensationConfig(
        update_period_ms=10.0,
        safety=SafetyConfig(**overrides),  # type: ignore[arg-type]
    )
    return CncSafetyGuards(config)


def _step(guards: CncSafetyGuards, target_x: float, now_sec: float | None) -> float:
    before = guards.last_offset.x
    decision = guards.evaluate(
        CncUserOffset(target_x, 0.0, 0.0, 0.0, 0.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
        now_sec=now_sec,
    )
    return decision.offset.x - before


def test_default_ceiling_matches_vmax_times_nominal_tick() -> None:
    """Default config must behave exactly as it did before the tick was measured."""
    guards = _guards()
    assert _step(guards, 5.0, None) == pytest.approx(0.6)


def test_slow_tick_does_not_enlarge_the_step() -> None:
    """A 60 ms tick would let an honest 60 mm/s vmax authorise 3.6 mm; the ceiling caps it."""
    guards = _guards()
    _step(guards, 5.0, 0.0)
    assert guards.measured_period_sec == pytest.approx(0.010)
    step = _step(guards, 5.0, 0.060)
    assert guards.measured_period_sec == pytest.approx(0.060)
    assert step == pytest.approx(0.6)


def test_fast_tick_lets_the_velocity_limit_bind() -> None:
    """Below the nominal tick, vmax is tighter than the ceiling and takes over."""
    guards = _guards()
    _step(guards, 5.0, 0.0)
    step = _step(guards, 5.0, 0.005)
    assert step == pytest.approx(0.3)


def test_stalled_loop_is_clamped() -> None:
    guards = _guards()
    _step(guards, 5.0, 0.0)
    _step(guards, 5.0, 10.0)
    assert guards.measured_period_sec == pytest.approx(MAX_PERIOD_SEC)


def test_backwards_or_zero_tick_is_ignored() -> None:
    guards = _guards()
    _step(guards, 5.0, 1.0)
    _step(guards, 5.0, 1.0)
    assert guards.measured_period_sec == pytest.approx(0.010)
    _step(guards, 5.0, 0.5)
    assert guards.measured_period_sec == pytest.approx(0.010)


def test_explicit_ceiling_overrides_the_derived_default() -> None:
    guards = _guards(max_step_mm=(0.2, 0.2, 0.2))
    assert _step(guards, 5.0, None) == pytest.approx(0.2)


def test_explicit_ceiling_still_loses_to_a_tighter_velocity_limit() -> None:
    guards = _guards(max_step_mm=(5.0, 5.0, 5.0))
    _step(guards, 50.0, 0.0)
    # 60 mm/s over a 10 ms tick is 0.6 mm, well below the 5 mm ceiling.
    assert _step(guards, 50.0, 0.010) == pytest.approx(0.6)


def test_rotation_ceiling_is_derived_per_axis() -> None:
    guards = _guards()
    decision = guards.evaluate(
        CncUserOffset(0.0, 0.0, 0.0, 5.0, 5.0),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.offset.b == pytest.approx(0.25)
    assert decision.offset.c == pytest.approx(0.25)
