from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cnc_config import BcAxisSign, CncCompensationConfig, MachinePose, OffsetDeadbandConfig
from .cnc_kinematics import (
    bc_from_camera_rvec_delta,
    bc_from_euler_delta,
    bc_from_normal,
    camera_rvec_delta_degrees,
    euler_delta_machine_degrees,
    normal_from_pose_rotation,
    nozzle_tip,
    rotation_delta_matrix,
    rotation_matrix_in_machine_frame,
    rvec_to_matrix,
    solve_bc_delta_pose_aware,
    solve_xyz_for_tip_delta,
    tip_rotation_delta_from_head,
)
from .types import HeadPose

_TRANSLATION_AXES = ("x", "y", "z")
_ROTATION_AXES = ("b", "c")
_ALL_AXES = _TRANSLATION_AXES + _ROTATION_AXES

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
    machine_pose: MachinePose | None = None
    ready: bool = True


class OffsetZeroLatch:
    """Translation uses vector-norm hysteresis; B/C stay per-axis."""

    def __init__(self, config: OffsetDeadbandConfig) -> None:
        self.config = config
        self._translation_latched = True
        self._latched = {axis: True for axis in _ROTATION_AXES}

    def reset(self) -> None:
        self._translation_latched = True
        for axis in self._latched:
            self._latched[axis] = True

    def apply(self, offset: CncUserOffset) -> CncUserOffset:
        if not self.config.enabled:
            return offset
        x, y, z = self._apply_translation_vector(offset.x, offset.y, offset.z)
        values: dict[str, float] = {"x": x, "y": y, "z": z}
        for axis in _ROTATION_AXES:
            values[axis] = self._apply_axis(axis, getattr(offset, axis), mm=False)
        return CncUserOffset(
            x=values["x"],
            y=values["y"],
            z=values["z"],
            b=values["b"],
            c=values["c"],
        )

    def _apply_translation_vector(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        norm = float(np.linalg.norm([x, y, z]))
        exit_mm = self.config.exit_translation_mm
        enter_mm = self.config.enter_translation_mm
        if self._translation_latched:
            if norm > exit_mm:
                self._translation_latched = False
                return float(x), float(y), float(z)
            return 0.0, 0.0, 0.0
        if norm < enter_mm:
            self._translation_latched = True
            return 0.0, 0.0, 0.0
        return float(x), float(y), float(z)

    def _apply_axis(self, axis: str, value: float, *, mm: bool) -> float:
        enter = self.config.enter_translation_mm if mm else self.config.enter_rotation_deg
        exit_ = self.config.exit_translation_mm if mm else self.config.exit_rotation_deg
        if self._latched[axis]:
            if abs(value) > exit_:
                self._latched[axis] = False
                return float(value)
            return 0.0
        if abs(value) < enter:
            self._latched[axis] = True
            return 0.0
        return float(value)


def offset_is_zero(offset: CncUserOffset) -> bool:
    return offset == CncUserOffset.zero()


def offset_within_exit_band(offset: CncUserOffset, config: OffsetDeadbandConfig) -> bool:
    if not config.enabled:
        return offset_is_zero(offset)
    for axis in _TRANSLATION_AXES:
        if abs(getattr(offset, axis)) > config.exit_translation_mm:
            return False
    for axis in _ROTATION_AXES:
        if abs(getattr(offset, axis)) > config.exit_rotation_deg:
            return False
    return True


class CncOffsetEncoder:
    def __init__(self, config: CncCompensationConfig) -> None:
        self.config = config
        self._baseline: BaselineState | None = None
        self._zero_latch = OffsetZeroLatch(config.offset_deadband)

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None and self._baseline.ready

    def reset_baseline(self) -> None:
        self._baseline = None
        self._zero_latch.reset()

    def capture_baseline(
        self,
        pose: HeadPose,
        *,
        machine_pose: MachinePose | None = None,
    ) -> None:
        rmat = rvec_to_matrix(pose.rotation_vector)
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        n = normal_from_pose_rotation(rmat)
        if np.linalg.norm(n) < 1e-12:
            n = np.asarray(self.config.reference_normal, dtype=np.float64).reshape(3)
            n = n / max(np.linalg.norm(n), 1e-12)
        resolved_pose = machine_pose or self.config.machine_pose
        self._baseline = BaselineState(
            translation_mm=t.copy(),
            rotation_matrix=rmat.copy(),
            normal=n.copy(),
            machine_pose=resolved_pose,
            ready=True,
        )
        self._zero_latch.reset()

    def _head_delta(self, pose: HeadPose) -> tuple[np.ndarray, np.ndarray]:
        if self._baseline is None:
            raise RuntimeError("baseline not captured")
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        r_current = rvec_to_matrix(pose.rotation_vector)
        d_t_cam = t - self._baseline.translation_mm
        d_t_machine = self.config.camera_to_machine_rotation @ d_t_cam
        d_r = rotation_delta_matrix(self._baseline.rotation_matrix, r_current)
        return d_t_machine, d_r

    def _head_normal_machine(self, rmat: np.ndarray) -> np.ndarray:
        n_cam = normal_from_pose_rotation(rmat)
        n = self.config.camera_to_machine_rotation @ n_cam
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            ref = np.asarray(self.config.reference_normal, dtype=np.float64).reshape(3)
            ref_norm = float(np.linalg.norm(ref))
            if ref_norm < 1e-12:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return ref / ref_norm
        return n / norm

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

    def _encode_bc(
        self,
        *,
        r_current: np.ndarray,
        machine_pose: MachinePose | None = None,
    ) -> tuple[float, float]:
        sign = self.config.sign()
        bc_sign = self.config.bc_axis_sign
        if self._baseline is None:
            return 0.0, 0.0
        if self.config.bc_mode == "tool_normal_ik":
            pose_machine = machine_pose or self.config.machine_pose
            if pose_machine is None:
                return self._encode_bc_camera_rvec(
                    r_current=r_current,
                    sign=sign,
                    bc_sign=bc_sign,
                )
            mapping = self.config.bc_camera_rvec_mapping
            db, dc = solve_bc_delta_pose_aware(
                r_current,
                self._baseline.rotation_matrix,
                self.config.camera_to_machine_rotation,
                pose_machine.b_deg,
                pose_machine.c_deg,
                self.config.machine,
                b_cam_axis=mapping.b_axis,
                c_cam_axis=mapping.c_axis,
                b_sign=bc_sign.b,
                c_sign=bc_sign.c,
                follow_sign=sign,
            )
            return db, dc
        if self.config.bc_mode == "camera_rvec":
            return self._encode_bc_camera_rvec(
                r_current=r_current,
                sign=sign,
                bc_sign=bc_sign,
            )
        if self.config.bc_mode == "euler_machine":
            pitch_d, yaw_d, roll_d = euler_delta_machine_degrees(
                r_current,
                self._baseline.rotation_matrix,
                self.config.camera_to_machine_rotation,
            )
            mapping = self.config.bc_euler_mapping
            return bc_from_euler_delta(
                pitch_d,
                yaw_d,
                roll_d,
                b_axis=mapping.b_axis,
                c_axis=mapping.c_axis,
                b_sign=bc_sign.b,
                c_sign=bc_sign.c,
                follow_sign=sign,
            )
        n_ref = self._head_normal_machine(self._baseline.rotation_matrix)
        n_new = self._head_normal_machine(r_current)
        b_ref, c_ref = bc_from_normal(n_ref)
        b_new, c_new = bc_from_normal(n_new)
        return (
            sign * bc_sign.b * float(b_new - b_ref),
            sign * bc_sign.c * float(c_new - c_ref),
        )

    def _encode_bc_camera_rvec(
        self,
        *,
        r_current: np.ndarray,
        sign: float,
        bc_sign: BcAxisSign,
    ) -> tuple[float, float]:
        rx_d, ry_d, rz_d = camera_rvec_delta_degrees(
            r_current,
            self._baseline.rotation_matrix,  # type: ignore[union-attr]
        )
        mapping = self.config.bc_camera_rvec_mapping
        return bc_from_camera_rvec_delta(
            rx_d,
            ry_d,
            rz_d,
            b_axis=mapping.b_axis,
            c_axis=mapping.c_axis,
            b_sign=bc_sign.b,
            c_sign=bc_sign.c,
            follow_sign=sign,
        )

    def _encode_decoupled(
        self,
        d_t_machine: np.ndarray,
        d_r: np.ndarray,
        *,
        r_current: np.ndarray,
        machine_pose: MachinePose | None = None,
    ) -> CncUserOffset:
        sign = self.config.sign()
        d_t = sign * d_t_machine
        b_off, c_off = self._encode_bc(
            r_current=r_current,
            machine_pose=machine_pose,
        )
        return CncUserOffset(
            x=float(d_t[0]),
            y=float(d_t[1]),
            z=float(d_t[2]),
            b=float(b_off),
            c=float(c_off),
        )

    def _encode_kinematic(
        self,
        d_t_machine: np.ndarray,
        d_r: np.ndarray,
        machine_pose: MachinePose,
        *,
        r_current: np.ndarray,
        head_translation_cam: np.ndarray,
    ) -> CncUserOffset:
        decoupled = self._encode_decoupled(
            d_t_machine,
            d_r,
            r_current=r_current,
            machine_pose=machine_pose,
        )
        sign = self.config.sign()
        b_cmd = machine_pose.b_deg + decoupled.b
        c_cmd = machine_pose.c_deg + decoupled.c
        r_delta_m = rotation_matrix_in_machine_frame(
            d_r,
            self.config.camera_to_machine_rotation,
        )
        head_center_machine = self.config.camera_to_machine_rotation @ np.asarray(
            head_translation_cam, dtype=np.float64
        ).reshape(3)
        tip = nozzle_tip(
            machine_pose.x,
            machine_pose.y,
            machine_pose.z,
            b_cmd,
            c_cmd,
            self.config.machine,
        )
        tip_rot_delta = tip_rotation_delta_from_head(
            r_delta_m,
            head_center_machine,
            tip,
        )
        tip_delta = sign * d_t_machine + sign * tip_rot_delta
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

    def head_delta_machine_mm(self, pose: HeadPose) -> np.ndarray | None:
        if self._baseline is None:
            return None
        d_t_machine, _ = self._head_delta(pose)
        return d_t_machine

    def encode(
        self,
        pose: HeadPose,
        *,
        machine_pose: MachinePose | None = None,
    ) -> CncUserOffset:
        if self._baseline is None:
            return CncUserOffset.zero()
        d_t_machine, d_r = self._head_delta(pose)
        r_current = rvec_to_matrix(pose.rotation_vector)
        pose_machine = machine_pose or self.config.machine_pose
        head_translation_cam = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        if self.config.solver == "kinematic" and pose_machine is not None:
            raw = self._encode_kinematic(
                d_t_machine,
                d_r,
                pose_machine,
                r_current=r_current,
                head_translation_cam=head_translation_cam,
            )
        else:
            raw = self._encode_decoupled(
                d_t_machine,
                d_r,
                r_current=r_current,
                machine_pose=pose_machine,
            )
        return self._zero_latch.apply(self._apply_limits(raw))

    def head_speed_mm_s(self, pose: HeadPose, dt_sec: float) -> float:
        if self._baseline is None or dt_sec <= 0.0:
            return 0.0
        t = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
        d_t_cam = t - self._baseline.translation_mm
        d_t_machine = self.config.camera_to_machine_rotation @ d_t_cam
        return float(np.linalg.norm(d_t_machine) / dt_sec)
