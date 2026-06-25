from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cnc_config import CncCompensationConfig, MachinePose
from .cnc_kinematics import (
    bc_from_normal,
    normal_from_pose_rotation,
    rotation_delta_matrix,
    rvec_to_matrix,
    solve_xyz_for_tip_delta,
)
from .types import HeadPose

@dataclass(frozen=True)
class CncUserOffset:
    x: float
    y: float
    z: float
    b: float
    c: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.x, self.y, self.z, self.b, self.c

    @classmethod
    def zero(cls) -> CncUserOffset:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class BaselineState:
    translation_mm: np.ndarray
    rotation_matrix: np.ndarray
    normal: np.ndarray
    ready: bool = True


class CncOffsetEncoder:
    def __init__(self, config: CncCompensationConfig) -> None:
        self.config = config
        self._baseline: BaselineState | None = None

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None and self._baseline.ready

    def reset_baseline(self) -> None:
        self._baseline = None

    def capture_baseline(self, pose: HeadPose) -> None:
        rmat = rvec_to_matrix(pose.rotation_vector)
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        n = normal_from_pose_rotation(rmat)
        if np.linalg.norm(n) < 1e-12:
            n = np.asarray(self.config.reference_normal, dtype=np.float64).reshape(3)
            n = n / max(np.linalg.norm(n), 1e-12)
        self._baseline = BaselineState(
            translation_mm=t.copy(),
            rotation_matrix=rmat.copy(),
            normal=n.copy(),
            ready=True,
        )

    def _head_delta(self, pose: HeadPose) -> tuple[np.ndarray, np.ndarray]:
        if self._baseline is None:
            raise RuntimeError("baseline not captured")
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        r_current = rvec_to_matrix(pose.rotation_vector)
        d_t_cam = t - self._baseline.translation_mm
        d_t_machine = self.config.camera_to_machine_rotation @ d_t_cam
        d_r = rotation_delta_matrix(self._baseline.rotation_matrix, r_current)
        return d_t_machine, d_r

    def _clamp_axis(self, name: str, value: float, limits_map: dict[str, tuple[float, float]]) -> float:
        lo, hi = limits_map[name]
        return float(np.clip(value, lo, hi))

    def _apply_limits(self, offset: CncUserOffset) -> CncUserOffset:
        lim = self.config.axis_limits.as_dict()
        scale_mm = self.config.mm_to_axis_unit
        scale_deg = self.config.deg_to_axis_unit
        return CncUserOffset(
            x=self._clamp_axis("x", offset.x / scale_mm, lim) * scale_mm,
            y=self._clamp_axis("y", offset.y / scale_mm, lim) * scale_mm,
            z=self._clamp_axis("z", offset.z / scale_mm, lim) * scale_mm,
            b=self._clamp_axis("b", offset.b / scale_deg, lim) * scale_deg,
            c=self._clamp_axis("c", offset.c / scale_deg, lim) * scale_deg,
        )

    def _encode_decoupled(
        self,
        d_t_machine: np.ndarray,
        d_r: np.ndarray,
    ) -> CncUserOffset:
        sign = self.config.sign()
        d_t = sign * d_t_machine
        n_ref = self._baseline.normal if self._baseline is not None else np.array([0.0, 0.0, 1.0])
        n_new = d_r @ n_ref
        b_ref, c_ref = bc_from_normal(n_ref)
        b_new, c_new = bc_from_normal(n_new)
        return CncUserOffset(
            x=float(d_t[0]),
            y=float(d_t[1]),
            z=float(d_t[2]),
            b=sign * float(b_new - b_ref),
            c=sign * float(c_new - c_ref),
        )

    def _encode_kinematic(
        self,
        d_t_machine: np.ndarray,
        d_r: np.ndarray,
        machine_pose: MachinePose,
    ) -> CncUserOffset:
        decoupled = self._encode_decoupled(d_t_machine, d_r)
        sign = self.config.sign()
        b_cmd = machine_pose.b_deg + decoupled.b
        c_cmd = machine_pose.c_deg + decoupled.c
        tip_delta = sign * d_t_machine
        x_adj, y_adj, z_adj = solve_xyz_for_tip_delta(
            machine_pose.x,
            machine_pose.y,
            machine_pose.z,
            b_cmd,
            c_cmd,
            tip_delta,
            self.config.machine,
        )
        return CncUserOffset(
            x=x_adj - machine_pose.x,
            y=y_adj - machine_pose.y,
            z=z_adj - machine_pose.z,
            b=decoupled.b,
            c=decoupled.c,
        )

    def encode(
        self,
        pose: HeadPose,
        *,
        machine_pose: MachinePose | None = None,
    ) -> CncUserOffset:
        if self._baseline is None:
            return CncUserOffset.zero()
        d_t_machine, d_r = self._head_delta(pose)
        pose_machine = machine_pose or self.config.machine_pose
        if self.config.solver == "kinematic" and pose_machine is not None:
            raw = self._encode_kinematic(d_t_machine, d_r, pose_machine)
        else:
            raw = self._encode_decoupled(d_t_machine, d_r)
        return self._apply_limits(raw)

    def head_speed_mm_s(self, pose: HeadPose, dt_sec: float) -> float:
        if self._baseline is None or dt_sec <= 0.0:
            return 0.0
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        d_t_cam = t - self._baseline.translation_mm
        d_t_machine = self.config.camera_to_machine_rotation @ d_t_cam
        return float(np.linalg.norm(d_t_machine) / dt_sec)
