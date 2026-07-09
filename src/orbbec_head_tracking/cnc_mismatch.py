from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .cnc_config import AxisLimits, CncCompensationConfig, MachinePose, MismatchConfig
from .cnc_kinematics import nozzle_tip
from .cnc_offset_encoder import CncUserOffset, offset_is_zero

MismatchMode = Literal["disabled", "pass", "pi", "snap", "zero", "track"]


@dataclass(frozen=True)
class MachineBaseline:
    pose: MachinePose
    tip_mm: np.ndarray


@dataclass(frozen=True)
class MismatchReport:
    offset_error: CncUserOffset
    error_norm_mm: float
    error_norm_deg: float
    mode: MismatchMode
    fault: bool = False
    tip_error_mm: np.ndarray | None = None



def offset_add(a: CncUserOffset, b: CncUserOffset) -> CncUserOffset:
    return CncUserOffset(
        x=a.x + b.x,
        y=a.y + b.y,
        z=a.z + b.z,
        b=a.b + b.b,
        c=a.c + b.c,
    )


def offset_sub(a: CncUserOffset, b: CncUserOffset) -> CncUserOffset:
    return CncUserOffset(
        x=a.x - b.x,
        y=a.y - b.y,
        z=a.z - b.z,
        b=a.b - b.b,
        c=a.c - b.c,
    )


def offset_scale(offset: CncUserOffset, factor: float) -> CncUserOffset:
    return CncUserOffset(
        x=offset.x * factor,
        y=offset.y * factor,
        z=offset.z * factor,
        b=offset.b * factor,
        c=offset.c * factor,
    )


def offset_translation_norm_mm(offset: CncUserOffset) -> float:
    return float(np.linalg.norm([offset.x, offset.y, offset.z]))


def offset_rotation_norm_deg(offset: CncUserOffset) -> float:
    return float(np.linalg.norm([offset.b, offset.c]))


def clamp_offset(offset: CncUserOffset, limits: AxisLimits) -> CncUserOffset:
    lim = limits.as_dict()
    def clamp_axis(name: str, value: float) -> float:
        lo, hi = lim[name]
        return float(np.clip(value, lo, hi))

    return CncUserOffset(
        x=clamp_axis("x", offset.x),
        y=clamp_axis("y", offset.y),
        z=clamp_axis("z", offset.z),
        b=clamp_axis("b", offset.b),
        c=clamp_axis("c", offset.c),
    )


def clamp_integral(integral: CncUserOffset, config: MismatchConfig) -> CncUserOffset:
    lim_mm = config.integral_limit_mm
    lim_deg = config.integral_limit_deg

    def clamp_mm(value: float) -> float:
        return float(np.clip(value, -lim_mm, lim_mm))

    def clamp_deg(value: float) -> float:
        return float(np.clip(value, -lim_deg, lim_deg))

    return CncUserOffset(
        x=clamp_mm(integral.x),
        y=clamp_mm(integral.y),
        z=clamp_mm(integral.z),
        b=clamp_deg(integral.b),
        c=clamp_deg(integral.c),
    )


class CncMismatchTracker:
    def __init__(self, config: CncCompensationConfig) -> None:
        self.config = config
        self.mismatch: MismatchConfig = config.mismatch
        self._baseline: MachineBaseline | None = None
        self._integral = CncUserOffset.zero()
        self._last_report: MismatchReport | None = None
        self._suppress_boost_ticks = 0

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    @property
    def last_report(self) -> MismatchReport | None:
        return self._last_report

    def reset(self) -> None:
        self._integral = CncUserOffset.zero()
        self._last_report = None
        self._suppress_boost_ticks = 0

    def notify_command_hold(self, reason: str = "") -> None:
        ticks = self.mismatch.recovery_ticks_after_hold
        self._suppress_boost_ticks = max(self._suppress_boost_ticks, ticks)

    def capture_baseline(self, machine_pose: MachinePose) -> None:
        machine = self.config.machine
        tip = nozzle_tip(
            machine_pose.x,
            machine_pose.y,
            machine_pose.z,
            machine_pose.b_deg,
            machine_pose.c_deg,
            machine,
        )
        self._baseline = MachineBaseline(pose=machine_pose, tip_mm=tip.copy())
        self.reset()

    def compute_error(self, required: CncUserOffset, sent: CncUserOffset) -> CncUserOffset:
        return offset_sub(required, sent)

    def compute_tip_error(
        self,
        sent: CncUserOffset,
        d_t_machine: np.ndarray,
    ) -> np.ndarray | None:
        if self._baseline is None:
            return None
        machine = self.config.machine
        baseline = self._baseline
        tip_cmd = nozzle_tip(
            baseline.pose.x + sent.x,
            baseline.pose.y + sent.y,
            baseline.pose.z + sent.z,
            baseline.pose.b_deg + sent.b,
            baseline.pose.c_deg + sent.c,
            machine,
        )
        tip_req = baseline.tip_mm + self.config.sign() * np.asarray(d_t_machine, dtype=float).reshape(3)
        return tip_req - tip_cmd

    def target(
        self,
        required: CncUserOffset,
        sent: CncUserOffset,
        *,
        dt_sec: float,
        head_speed_mm_s: float,
        d_t_machine: np.ndarray | None = None,
        preserve_sent: bool = False,
    ) -> tuple[CncUserOffset, MismatchReport]:
        error = self.compute_error(required, sent)
        error_norm_mm = offset_translation_norm_mm(error)
        error_norm_deg = offset_rotation_norm_deg(error)
        tip_error = None
        if d_t_machine is not None and self._baseline is not None:
            tip_error = self.compute_tip_error(sent, d_t_machine)

        fault = (
            self.mismatch.fault_error_mm is not None
            and error_norm_mm > float(self.mismatch.fault_error_mm)
        )

        if not self.mismatch.enabled:
            report = MismatchReport(
                offset_error=error,
                error_norm_mm=error_norm_mm,
                error_norm_deg=error_norm_deg,
                mode="disabled",
                fault=fault,
                tip_error_mm=tip_error,
            )
            self._last_report = report
            return required, report

        if offset_is_zero(required):
            self._integral = CncUserOffset.zero()
            self._suppress_boost_ticks = 0
            if preserve_sent and not offset_is_zero(sent):
                report = MismatchReport(
                    offset_error=error,
                    error_norm_mm=error_norm_mm,
                    error_norm_deg=error_norm_deg,
                    mode="preserve_sent",
                    fault=fault,
                    tip_error_mm=tip_error,
                )
                self._last_report = report
                return sent, report
            report = MismatchReport(
                offset_error=error,
                error_norm_mm=error_norm_mm,
                error_norm_deg=error_norm_deg,
                mode="zero",
                fault=fault,
                tip_error_mm=tip_error,
            )
            self._last_report = report
            return CncUserOffset.zero(), report

        suppress_boost = self._suppress_boost_ticks > 0
        if self._suppress_boost_ticks > 0:
            self._suppress_boost_ticks -= 1

        cfg = self.mismatch
        stable = head_speed_mm_s <= cfg.snap_head_speed_mm_s
        if cfg.snap_enabled and stable and error_norm_mm >= cfg.snap_error_mm:
            report = MismatchReport(
                offset_error=error,
                error_norm_mm=error_norm_mm,
                error_norm_deg=error_norm_deg,
                mode="snap",
                fault=fault,
                tip_error_mm=tip_error,
            )
            self._last_report = report
            return clamp_offset(required, self.config.axis_limits), report

        use_boost = not suppress_boost and (cfg.kp > 0.0 or cfg.ki > 0.0)
        if not use_boost:
            report = MismatchReport(
                offset_error=error,
                error_norm_mm=error_norm_mm,
                error_norm_deg=error_norm_deg,
                mode="track",
                fault=fault,
                tip_error_mm=tip_error,
            )
            self._last_report = report
            return clamp_offset(required, self.config.axis_limits), report

        if cfg.ki > 0.0 and dt_sec > 0.0:
            self._integral = clamp_integral(
                offset_add(self._integral, offset_scale(error, dt_sec)),
                cfg,
            )

        correction = offset_scale(error, cfg.kp) if cfg.kp > 0.0 else CncUserOffset.zero()
        if cfg.ki > 0.0:
            correction = offset_add(correction, offset_scale(self._integral, cfg.ki))

        boosted = clamp_offset(
            offset_add(required, correction),
            self.config.axis_limits,
        )
        report = MismatchReport(
            offset_error=error,
            error_norm_mm=error_norm_mm,
            error_norm_deg=error_norm_deg,
            mode="pi",
            fault=fault,
            tip_error_mm=tip_error,
        )
        self._last_report = report
        return boosted, report
