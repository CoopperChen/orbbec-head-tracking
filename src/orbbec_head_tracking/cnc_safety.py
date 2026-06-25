from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cnc_config import CncCompensationConfig, SafetyConfig
from .cnc_offset_encoder import CncUserOffset

SafetyAction = Literal["pass", "zero", "hold_last"]


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


class CncSafetyGuards:
    def __init__(self, config: CncCompensationConfig) -> None:
        self.config = config
        self.safety: SafetyConfig = config.safety
        self._state = SafetyState()
        self._period_sec = max(config.update_period_ms, 0.001) / 1000.0

    @property
    def last_offset(self) -> CncUserOffset:
        return self._state.last_offset

    def reset(self) -> None:
        self._state = SafetyState()

    def _max_step(self, axis: str) -> float:
        if axis in ("x", "y", "z"):
            idx = {"x": 0, "y": 1, "z": 2}[axis]
            return self.safety.vmax_mm_s[idx] * self._period_sec
        idx = {"b": 0, "c": 1}[axis]
        return self.safety.vmax_deg_s[idx] * self._period_sec

    def _rate_limit(self, proposed: CncUserOffset) -> CncUserOffset:
        prev = self._state.last_offset
        def step(cur: float, prev_val: float, axis: str) -> float:
            max_d = self._max_step(axis)
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

    def _reject_spike(self, proposed: CncUserOffset) -> bool:
        prev = self._state.last_proposed
        if prev is None:
            return False
        limits = {
            "x": self._max_step("x") * 3.0,
            "y": self._max_step("y") * 3.0,
            "z": self._max_step("z") * 3.0,
            "b": self._max_step("b") * 3.0,
            "c": self._max_step("c") * 3.0,
        }
        deltas = (
            abs(proposed.x - prev.x),
            abs(proposed.y - prev.y),
            abs(proposed.z - prev.z),
            abs(proposed.b - prev.b),
            abs(proposed.c - prev.c),
        )
        keys = ("x", "y", "z", "b", "c")
        return any(d > limits[k] for d, k in zip(deltas, keys, strict=True))

    def _clear_motion_state(self) -> None:
        self._state.last_offset = CncUserOffset.zero()
        self._state.last_proposed = None

    def evaluate(
        self,
        proposed: CncUserOffset,
        *,
        tracking_ok: bool,
        confidence: float,
        baseline_ready: bool,
        head_speed_mm_s: float,
        link_ok: bool = True,
    ) -> SafetyDecision:
        if not link_ok:
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "udp_link_fault")

        if self.safety.require_baseline_before_stream and not baseline_ready:
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "baseline_not_ready")

        if not tracking_ok:
            if self.safety.on_tracking_loss == "hold_last":
                return SafetyDecision("hold_last", self._state.last_offset, "tracking_lost")
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "tracking_lost")

        if confidence < self.safety.min_confidence:
            if self.safety.on_low_confidence == "hold_last":
                return SafetyDecision("hold_last", self._state.last_offset, "low_confidence")
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "low_confidence")

        if head_speed_mm_s > self.safety.max_head_speed_mm_s:
            self._clear_motion_state()
            return SafetyDecision("zero", CncUserOffset.zero(), "head_speed_exceeded")

        if self._reject_spike(proposed):
            return SafetyDecision("hold_last", self._state.last_offset, "spike_rejected")

        self._state.last_proposed = proposed
        limited = self._rate_limit(proposed)
        self._state.last_offset = limited
        return SafetyDecision("pass", limited, "")

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
        return float(np.linalg.norm(delta) / dt)
