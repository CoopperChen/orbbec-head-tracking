from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cnc_config import CncCompensationConfig, SafetyConfig
from .cnc_offset_encoder import CncUserOffset, offset_is_zero, offset_within_exit_band

SafetyAction = Literal["pass", "zero", "hold_last"]

_TRANSLATION_AXES = ("x", "y", "z")
_ROTATION_AXES = ("b", "c")
_ALL_AXES = _TRANSLATION_AXES + _ROTATION_AXES


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    offset: CncUserOffset
    reason: str = ""


@dataclass
class SafetyState:
    last_offset: CncUserOffset = CncUserOffset.zero()
    last_proposed: CncUserOffset | None = None
    last_pose_translation: np.ndarray | None = None
    last_pose_time_sec: float | None = None
    filtered_head_speed_mm_s: float = 0.0
    head_speed_exceed_count: int = 0
    recovery_ticks: int = 0


class CncSafetyGuards:
    def __init__(self, config: CncCompensationConfig) -> None:
        self.config = config
        self.safety: SafetyConfig = config.safety
        self._state = SafetyState()
        self._period_sec = max(config.update_period_ms, 0.001) / 1000.0

    @property
    def last_offset(self) -> CncUserOffset:
        return self._state.last_offset

    @property
    def recovery_ticks(self) -> int:
        return self._state.recovery_ticks

    def reset(self) -> None:
        self._state = SafetyState()

    def notify_hold(self, reason: str) -> None:
        """Arm a gentle recovery ramp after hold_last (spike, tracking loss, etc.)."""
        ticks = max(0, int(self.safety.recovery_ticks_after_hold))
        self._state.recovery_ticks = max(self._state.recovery_ticks, ticks)
        if reason in ("tracking_lost", "low_confidence"):
            self._state.last_proposed = CncUserOffset(
                x=self._state.last_offset.x,
                y=self._state.last_offset.y,
                z=self._state.last_offset.z,
                b=self._state.last_offset.b,
                c=self._state.last_offset.c,
            )

    def _max_step(self, axis: str) -> float:
        if axis in _TRANSLATION_AXES:
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            return self.safety.vmax_mm_s[idx] * self._period_sec
        idx = {"b": 0, "c": 1}[axis]
        return self.safety.vmax_deg_s[idx] * self._period_sec

    def _axis_limits(self, axis: str) -> tuple[float, float]:
        return self.config.axis_limits.as_dict()[axis]

    def _axis_at_limit(self, value: float, axis: str) -> bool:
        lo, hi = self._axis_limits(axis)
        eps = 0.15 if axis in _TRANSLATION_AXES else 0.25
        return value <= lo + eps or value >= hi - eps

    def _axis_value(self, offset: CncUserOffset, axis: str) -> float:
        return float(getattr(offset, axis))

    def _offset_at_limit(self, offset: CncUserOffset) -> bool:
        return any(self._axis_at_limit(self._axis_value(offset, axis), axis) for axis in _ALL_AXES)

    def _rate_limit(self, proposed: CncUserOffset, *, catch_up: bool = False) -> CncUserOffset:
        prev = self._state.last_offset
        multiplier = self.safety.catch_up_multiplier if catch_up else 1.0

        def step(cur: float, prev_val: float, axis: str) -> float:
            max_d = self._max_step(axis)
            if catch_up and multiplier > 1.0:
                max_d *= multiplier
            delta = cur - prev_val
            delta = float(np.clip(delta, -max_d, max_d))
            return prev_val + delta

        return CncUserOffset(
            x=step(proposed.x, prev.x, "x"),
            y=step(proposed.y, prev.y, "y"),
            z=step(proposed.z, prev.z, "z"),
            b=step(proposed.b, prev.b, "b"),
            c=step(proposed.c, prev.c, "c"),
        )

    def _sanitise_target(self, proposed: CncUserOffset) -> CncUserOffset:
        """During recovery, never leap toward a limit-saturated vision target."""
        if self._state.recovery_ticks <= 0:
            return proposed
        prev = self._state.last_offset
        values: dict[str, float] = {}
        for axis in _ALL_AXES:
            cur = self._axis_value(proposed, axis)
            prev_val = self._axis_value(prev, axis)
            max_d = self._max_step(axis)
            if self._axis_at_limit(cur, axis) and abs(cur - prev_val) > max_d:
                cur = prev_val + float(np.clip(cur - prev_val, -max_d, max_d))
            else:
                cur = prev_val + float(np.clip(cur - prev_val, -max_d, max_d))
            values[axis] = cur
        return CncUserOffset(
            x=values["x"],
            y=values["y"],
            z=values["z"],
            b=values["b"],
            c=values["c"],
        )

    def _vision_stable(self, vision: CncUserOffset) -> bool:
        prev = self._state.last_proposed
        if prev is None:
            return True
        for axis in _ALL_AXES:
            limit = self._max_step(axis) * 2.0
            if abs(self._axis_value(vision, axis) - self._axis_value(prev, axis)) > limit:
                return False
        return True

    def _should_catch_up(self, proposed: CncUserOffset, vision: CncUserOffset) -> bool:
        if self._state.recovery_ticks > 0:
            return False
        if self.safety.catch_up_multiplier <= 1.0:
            return False
        if self._offset_at_limit(proposed):
            return False
        if not self._vision_stable(vision):
            return False
        sent = self._state.last_offset
        translation_lag = float(
            np.linalg.norm(
                [
                    proposed.x - sent.x,
                    proposed.y - sent.y,
                    proposed.z - sent.z,
                ]
            )
        )
        rotation_lag = float(
            np.linalg.norm([proposed.b - sent.b, proposed.c - sent.c])
        )
        if translation_lag >= self.safety.catch_up_error_mm:
            return True
        return rotation_lag >= self.safety.catch_up_error_deg

    def _proposed_near_sent(self, proposed: CncUserOffset) -> bool:
        sent = self._state.last_offset
        for axis in self.safety.spike_axes:
            if abs(self._axis_value(proposed, axis) - self._axis_value(sent, axis)) > self._max_step(axis):
                return False
        return True

    def _reject_spike(self, proposed: CncUserOffset) -> bool:
        if self._state.recovery_ticks > 0:
            return False
        if self.safety.spike_multiplier <= 0.0:
            return False
        prev = self._state.last_proposed
        if prev is None:
            return False
        if self._proposed_near_sent(proposed):
            return False
        limits = {
            axis: self._max_step(axis) * float(self.safety.spike_multiplier)
            for axis in _ALL_AXES
        }
        for axis in self.safety.spike_axes:
            cur = self._axis_value(proposed, axis)
            previous = self._axis_value(prev, axis)
            if abs(cur - previous) > limits[axis]:
                return True
        return False

    def _clear_motion_state(self) -> None:
        self._state.last_offset = CncUserOffset.zero()
        self._state.last_proposed = None
        self._state.head_speed_exceed_count = 0
        self._state.recovery_ticks = 0

    def _hold_last(self, reason: str) -> SafetyDecision:
        self.notify_hold(reason)
        return SafetyDecision("hold_last", self._state.last_offset, reason)

    def evaluate(
        self,
        proposed: CncUserOffset,
        *,
        spike_reference: CncUserOffset | None = None,
        tracking_ok: bool,
        confidence: float,
        baseline_ready: bool,
        head_speed_mm_s: float,
        link_ok: bool = True,
        snap: bool = False,
    ) -> SafetyDecision:
        if not link_ok:
            if not offset_is_zero(self._state.last_offset):
                return self._hold_last("udp_link_fault")
            return SafetyDecision("zero", CncUserOffset.zero(), "udp_link_fault")

        if self.safety.require_baseline_before_stream and not baseline_ready:
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "baseline_not_ready")

        if not tracking_ok:
            if self.safety.on_tracking_loss == "hold_last":
                return self._hold_last("tracking_lost")
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "tracking_lost")

        if confidence < self.safety.min_confidence:
            if self.safety.on_low_confidence == "hold_last":
                return self._hold_last("low_confidence")
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "low_confidence")

        if head_speed_mm_s > self.safety.max_head_speed_mm_s:
            self._state.head_speed_exceed_count += 1
        else:
            self._state.head_speed_exceed_count = 0

        if self._state.head_speed_exceed_count >= max(1, int(self.safety.head_speed_exceed_ticks)):
            if self.safety.on_head_speed_exceeded == "hold_last":
                return self._hold_last("head_speed_exceeded")
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "head_speed_exceeded")

        vision = spike_reference if spike_reference is not None else proposed
        if self._reject_spike(vision):
            self._state.last_proposed = vision
            return self._hold_last("spike_rejected")

        self._state.last_proposed = vision
        if self._state.recovery_ticks > 0 and offset_is_zero(proposed):
            held = self._state.last_offset
            self._state.recovery_ticks -= 1
            return SafetyDecision("pass", held, "recovery_hold")

        target = self._sanitise_target(proposed)
        zero_target = offset_is_zero(target)
        if zero_target:
            target = CncUserOffset.zero()
            if offset_within_exit_band(self._state.last_offset, self.config.offset_deadband):
                self._state.last_offset = CncUserOffset.zero()
                self._state.recovery_ticks = 0
                return SafetyDecision("pass", CncUserOffset.zero(), "")
        catch_up = self._should_catch_up(target, vision)
        limited = target if snap else self._rate_limit(target, catch_up=catch_up)
        self._state.last_offset = limited
        in_recovery = self._state.recovery_ticks > 0
        if in_recovery:
            self._state.recovery_ticks -= 1
        if snap:
            reason = "snap"
        elif zero_target:
            reason = "zero_ramp"
        elif catch_up:
            reason = "catch_up"
        elif in_recovery:
            reason = "recovery_ramp"
        else:
            reason = ""
        return SafetyDecision("pass", limited, reason)

    def update_pose_timing(self, translation_mm: np.ndarray, time_sec: float) -> None:
        self._state.last_pose_translation = np.asarray(translation_mm, dtype=float).reshape(3).copy()
        self._state.last_pose_time_sec = float(time_sec)

    def estimate_head_speed_mm_s(self, translation_mm: np.ndarray, time_sec: float) -> float:
        prev_t = self._state.last_pose_translation
        prev_time = self._state.last_pose_time_sec
        self.update_pose_timing(translation_mm, time_sec)
        if prev_t is None or prev_time is None:
            return 0.0
        dt = time_sec - prev_time
        if dt <= 0.0:
            return 0.0
        rot = self.config.camera_to_machine_rotation
        delta = rot @ (np.asarray(translation_mm, dtype=float).reshape(3) - prev_t)
        raw_speed = float(np.linalg.norm(delta) / dt)
        alpha = float(np.clip(self.safety.head_speed_filter_alpha, 0.0, 1.0))
        filtered = (1.0 - alpha) * self._state.filtered_head_speed_mm_s + alpha * raw_speed
        self._state.filtered_head_speed_mm_s = float(filtered)
        return self._state.filtered_head_speed_mm_s
