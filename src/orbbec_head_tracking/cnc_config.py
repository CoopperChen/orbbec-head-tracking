from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .cnc_kinematics import MachineConfig
from .cnc_protocol import MotorAxisMap, parse_motor_axis_map

OffsetMode = Literal["follow", "counter"]
SolverMode = Literal["kinematic", "decoupled"]
TrackingLossPolicy = Literal["zero_offsets", "hold_last"]
BcMode = Literal["tool_normal_ik", "camera_rvec", "euler_machine", "normal"]
BcEulerAxis = Literal["pitch", "yaw", "roll"]
BcCameraRvecAxis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class AxisLimits:
    x_mm: tuple[float, float] = (-25.0, 25.0)
    y_mm: tuple[float, float] = (-25.0, 25.0)
    z_mm: tuple[float, float] = (-25.0, 25.0)
    b_deg: tuple[float, float] = (-15.0, 15.0)
    c_deg: tuple[float, float] = (-15.0, 15.0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "x": self.x_mm,
            "y": self.y_mm,
            "z": self.z_mm,
            "b": self.b_deg,
            "c": self.c_deg,
        }


@dataclass(frozen=True)
class OffsetDeadbandConfig:
    enabled: bool = True
    enter_translation_mm: float = 0.6
    exit_translation_mm: float = 1.2
    enter_rotation_deg: float = 0.5
    exit_rotation_deg: float = 1.0


@dataclass(frozen=True)
class MismatchConfig:
    enabled: bool = True
    kp: float = 0.0
    ki: float = 0.0
    integral_limit_mm: float = 10.0
    integral_limit_deg: float = 10.0
    snap_enabled: bool = False
    snap_head_speed_mm_s: float = 5.0
    snap_error_mm: float = 2.0
    fault_error_mm: float | None = None
    recovery_ticks_after_hold: int = 10


@dataclass(frozen=True)
class SafetyConfig:
    min_confidence: float = 0.6
    on_tracking_loss: TrackingLossPolicy = "hold_last"
    on_low_confidence: TrackingLossPolicy = "hold_last"
    on_head_speed_exceeded: TrackingLossPolicy = "hold_last"
    max_head_speed_mm_s: float = 80.0
    head_speed_filter_alpha: float = 0.25
    head_speed_exceed_ticks: int = 3
    vmax_mm_s: tuple[float, float, float] = (60.0, 60.0, 30.0)
    vmax_deg_s: tuple[float, float] = (25.0, 25.0)
    require_baseline_before_stream: bool = True
    min_standoff_mm: float | None = None
    spike_multiplier: float = 10.0
    spike_axes: tuple[str, ...] = ("x", "y", "z")
    catch_up_multiplier: float = 3.0
    catch_up_error_mm: float = 0.5
    catch_up_error_deg: float = 0.3
    recovery_ticks_after_hold: int = 20


@dataclass(frozen=True)
class MachinePose:
    x: float
    y: float
    z: float
    b_deg: float
    c_deg: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> MachinePose:
        if len(values) != 5:
            raise ValueError("machine pose requires 5 values: X,Y,Z,B,C")
        return cls(
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
            b_deg=float(values[3]),
            c_deg=float(values[4]),
        )

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.x, self.y, self.z, self.b_deg, self.c_deg


@dataclass(frozen=True)
class BcAxisSign:
    """Per-axis sign for B/C compensation vs head rotation (HICON convention)."""

    b: float = -1.0
    c: float = 1.0


@dataclass(frozen=True)
class BcCameraRvecMapping:
    """Map OpenCV camera-frame rotation-vector components to B/C offsets."""

    b_axis: BcCameraRvecAxis = "z"
    c_axis: BcCameraRvecAxis = "y"


@dataclass(frozen=True)
class BcEulerMapping:
    """Map machine-frame head Euler components to B/C offsets."""

    b_axis: BcEulerAxis = "pitch"
    c_axis: BcEulerAxis = "roll"


@dataclass(frozen=True)
class WorkPoseUdpConfig:
    port: int | None = None
    bind_ip: str = "0.0.0.0"
    stale_ms: float = 500.0
    require_live: bool = False


@dataclass(frozen=True)
class CncCompensationConfig:
    machine: MachineConfig = field(default_factory=MachineConfig)
    camera_to_machine_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    axis_limits: AxisLimits = field(default_factory=AxisLimits)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mismatch: MismatchConfig = field(default_factory=MismatchConfig)
    offset_deadband: OffsetDeadbandConfig = field(default_factory=OffsetDeadbandConfig)
    mm_to_axis_unit: float = 1.0
    deg_to_axis_unit: float = 1.0
    solver: SolverMode = "decoupled"
    offset_mode: OffsetMode = "follow"
    bc_mode: BcMode = "tool_normal_ik"
    bc_camera_rvec_mapping: BcCameraRvecMapping = field(default_factory=BcCameraRvecMapping)
    bc_euler_mapping: BcEulerMapping = field(default_factory=BcEulerMapping)
    bc_axis_sign: BcAxisSign = field(default_factory=BcAxisSign)
    machine_pose: MachinePose | None = None
    work_pose_udp: WorkPoseUdpConfig = field(default_factory=WorkPoseUdpConfig)
    reference_normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    update_period_ms: float = 10.0
    motor_map: MotorAxisMap = field(default_factory=MotorAxisMap.dual_x)

    def sign(self) -> float:
        return 1.0 if self.offset_mode == "follow" else -1.0


def _parse_bc_axis_sign(raw: dict[str, Any] | object | None) -> BcAxisSign:
    if raw is None:
        return BcAxisSign()
    if not isinstance(raw, dict):
        raise ValueError("bc_axis_sign must be a mapping with optional b and c keys")
    return BcAxisSign(
        b=float(raw.get("b", -1.0)),
        c=float(raw.get("c", 1.0)),
    )


def _parse_bc_camera_rvec_mapping(raw: dict[str, Any] | object | None) -> BcCameraRvecMapping:
    if raw is None:
        return BcCameraRvecMapping()
    if not isinstance(raw, dict):
        raise ValueError("bc_camera_rvec_mapping must be a mapping with optional b and c keys")
    allowed = {"x", "y", "z"}
    b_axis = str(raw.get("b", "z")).lower()
    c_axis = str(raw.get("c", "y")).lower()
    if b_axis not in allowed or c_axis not in allowed:
        raise ValueError("bc_camera_rvec_mapping axes must be x, y, or z")
    return BcCameraRvecMapping(b_axis=b_axis, c_axis=c_axis)  # type: ignore[arg-type]


def _parse_bc_euler_mapping(raw: dict[str, Any] | object | None) -> BcEulerMapping:
    if raw is None:
        return BcEulerMapping()
    if not isinstance(raw, dict):
        raise ValueError("bc_euler_mapping must be a mapping with optional b and c keys")
    allowed = {"pitch", "yaw", "roll"}
    b_axis = str(raw.get("b", "pitch")).lower()
    c_axis = str(raw.get("c", "roll")).lower()
    if b_axis not in allowed or c_axis not in allowed:
        raise ValueError("bc_euler_mapping axes must be pitch, yaw, or roll")
    return BcEulerMapping(b_axis=b_axis, c_axis=c_axis)  # type: ignore[arg-type]


def _parse_work_pose_udp(raw: dict[str, Any] | object | None) -> WorkPoseUdpConfig:
    if raw is None:
        return WorkPoseUdpConfig()
    if not isinstance(raw, dict):
        raise ValueError("work_pose_udp must be a mapping")
    port_raw = raw.get("port")
    port = int(port_raw) if port_raw is not None else None
    return WorkPoseUdpConfig(
        port=port,
        bind_ip=str(raw.get("bind_ip", "0.0.0.0")),
        stale_ms=float(raw.get("stale_ms", 500.0)),
        require_live=bool(raw.get("require_live", False)),
    )


def _parse_limits(raw: dict[str, Any]) -> AxisLimits:
    def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
        val = raw.get(key, list(default))
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            return default
        return float(val[0]), float(val[1])

    return AxisLimits(
        x_mm=pair("x_mm", (-25.0, 25.0)),
        y_mm=pair("y_mm", (-25.0, 25.0)),
        z_mm=pair("z_mm", (-25.0, 25.0)),
        b_deg=pair("b_deg", (-15.0, 15.0)),
        c_deg=pair("c_deg", (-15.0, 15.0)),
    )


def _parse_offset_deadband(raw: dict[str, Any]) -> OffsetDeadbandConfig:
    return OffsetDeadbandConfig(
        enabled=bool(raw.get("enabled", True)),
        enter_translation_mm=float(raw.get("enter_translation_mm", 0.6)),
        exit_translation_mm=float(raw.get("exit_translation_mm", 1.2)),
        enter_rotation_deg=float(raw.get("enter_rotation_deg", 0.5)),
        exit_rotation_deg=float(raw.get("exit_rotation_deg", 1.0)),
    )


def _parse_mismatch(raw: dict[str, Any]) -> MismatchConfig:
    fault = raw.get("fault_error_mm")
    return MismatchConfig(
        enabled=bool(raw.get("enabled", True)),
        kp=float(raw.get("kp", 0.0)),
        ki=float(raw.get("ki", 0.0)),
        integral_limit_mm=float(raw.get("integral_limit_mm", 10.0)),
        integral_limit_deg=float(raw.get("integral_limit_deg", 10.0)),
        snap_enabled=bool(raw.get("snap_enabled", False)),
        snap_head_speed_mm_s=float(raw.get("snap_head_speed_mm_s", 5.0)),
        snap_error_mm=float(raw.get("snap_error_mm", 2.0)),
        fault_error_mm=float(fault) if fault is not None else None,
        recovery_ticks_after_hold=int(raw.get("recovery_ticks_after_hold", 10)),
    )


def _parse_safety(raw: dict[str, Any]) -> SafetyConfig:
    vmax_mm = raw.get("vmax_mm_s", {})
    vmax_deg = raw.get("vmax_deg_s", {})
    return SafetyConfig(
        min_confidence=float(raw.get("min_confidence", 0.6)),
        on_tracking_loss=str(raw.get("on_tracking_loss", "hold_last")),  # type: ignore[arg-type]
        on_low_confidence=str(raw.get("on_low_confidence", "hold_last")),  # type: ignore[arg-type]
        on_head_speed_exceeded=str(raw.get("on_head_speed_exceeded", "hold_last")),  # type: ignore[arg-type]
        max_head_speed_mm_s=float(raw.get("max_head_speed_mm_s", 80.0)),
        head_speed_filter_alpha=float(raw.get("head_speed_filter_alpha", 0.25)),
        head_speed_exceed_ticks=int(raw.get("head_speed_exceed_ticks", 3)),
        vmax_mm_s=(
            float(vmax_mm.get("x", 60.0)),
            float(vmax_mm.get("y", 60.0)),
            float(vmax_mm.get("z", 30.0)),
        ),
        vmax_deg_s=(
            float(vmax_deg.get("b", 25.0)),
            float(vmax_deg.get("c", 25.0)),
        ),
        require_baseline_before_stream=bool(raw.get("require_baseline_before_stream", True)),
        min_standoff_mm=(
            float(raw["min_standoff_mm"]) if raw.get("min_standoff_mm") is not None else None
        ),
        spike_multiplier=float(raw.get("spike_multiplier", 10.0)),
        spike_axes=_parse_spike_axes(raw.get("spike_axes", ["x", "y", "z"])),
        catch_up_multiplier=float(raw.get("catch_up_multiplier", 3.0)),
        catch_up_error_mm=float(raw.get("catch_up_error_mm", 0.5)),
        catch_up_error_deg=float(raw.get("catch_up_error_deg", 0.3)),
        recovery_ticks_after_hold=int(raw.get("recovery_ticks_after_hold", 20)),
    )


def _parse_spike_axes(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ("x", "y", "z")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("spike_axes must be a list of axis names")
    allowed = {"x", "y", "z", "b", "c"}
    axes = tuple(str(axis).lower() for axis in raw)
    unknown = [axis for axis in axes if axis not in allowed]
    if unknown:
        raise ValueError(f"unknown spike_axes: {unknown}")
    return axes


def _parse_rotation_matrix(raw: list[list[float]] | None) -> np.ndarray:
    if raw is None:
        return np.eye(3, dtype=np.float64)
    mat = np.asarray(raw, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError("camera_extrinsic.rotation must be 3x3")
    return mat


def load_compensation_config(path: str | Path) -> CncCompensationConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load .yaml calibration files; "
                "install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("calibration file must contain a mapping at top level")

    machine_raw = data.get("machine", {})
    machine = MachineConfig(
        a_mm=float(machine_raw.get("a_mm", 180.7)),
        d_mm=float(machine_raw.get("d_mm", 57.59)),
        gap_size_mm=float(machine_raw.get("gap_size_mm", 15.0)),
        calgap_z_mm=float(machine_raw.get("calgap_z_mm", 26.62)),
        c0_deg=float(machine_raw.get("c0_deg", 90.0)),
        b0_deg=float(machine_raw.get("b0_deg", 0.0)),
    )

    extrinsic = data.get("camera_extrinsic", {})
    rotation = _parse_rotation_matrix(extrinsic.get("rotation"))
    ref_normal = np.asarray(
        data.get("reference_normal", [0.0, 0.0, 1.0]),
        dtype=np.float64,
    ).reshape(3)

    machine_pose = None
    if "machine_pose" in data:
        machine_pose = MachinePose.from_sequence(data["machine_pose"])

    return CncCompensationConfig(
        machine=machine,
        camera_to_machine_rotation=rotation,
        axis_limits=_parse_limits(data.get("axis_limits", {})),
        safety=_parse_safety(data.get("safety", {})),
        mismatch=_parse_mismatch(data.get("mismatch", {})),
        offset_deadband=_parse_offset_deadband(data.get("offset_deadband", {})),
        mm_to_axis_unit=float(data.get("mm_to_axis_unit", 1.0)),
        deg_to_axis_unit=float(data.get("deg_to_axis_unit", 1.0)),
        solver=str(data.get("solver", "decoupled")),  # type: ignore[arg-type]
        offset_mode=str(data.get("offset_mode", "follow")),  # type: ignore[arg-type]
        bc_mode=str(data.get("bc_mode", "tool_normal_ik")),  # type: ignore[arg-type]
        bc_camera_rvec_mapping=_parse_bc_camera_rvec_mapping(data.get("bc_camera_rvec_mapping")),
        bc_euler_mapping=_parse_bc_euler_mapping(data.get("bc_euler_mapping")),
        bc_axis_sign=_parse_bc_axis_sign(data.get("bc_axis_sign")),
        machine_pose=machine_pose,
        work_pose_udp=_parse_work_pose_udp(data.get("work_pose_udp")),
        reference_normal=ref_normal,
        update_period_ms=float(data.get("update_period_ms", 10.0)),
        motor_map=parse_motor_axis_map(data.get("motor_map", "dual_x")),
    )
