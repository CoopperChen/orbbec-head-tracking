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
class SafetyConfig:
    min_confidence: float = 0.6
    on_tracking_loss: TrackingLossPolicy = "zero_offsets"
    on_low_confidence: TrackingLossPolicy = "zero_offsets"
    max_head_speed_mm_s: float = 80.0
    vmax_mm_s: tuple[float, float, float] = (30.0, 30.0, 15.0)
    vmax_deg_s: tuple[float, float] = (10.0, 10.0)
    require_baseline_before_stream: bool = True
    min_standoff_mm: float | None = None


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
class CncCompensationConfig:
    machine: MachineConfig = field(default_factory=MachineConfig)
    camera_to_machine_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    axis_limits: AxisLimits = field(default_factory=AxisLimits)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mm_to_axis_unit: float = 1.0
    deg_to_axis_unit: float = 1.0
    solver: SolverMode = "decoupled"
    offset_mode: OffsetMode = "follow"
    machine_pose: MachinePose | None = None
    reference_normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    update_period_ms: float = 10.0
    motor_map: MotorAxisMap = field(default_factory=MotorAxisMap.dual_x)

    def sign(self) -> float:
        return 1.0 if self.offset_mode == "follow" else -1.0


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


def _parse_safety(raw: dict[str, Any]) -> SafetyConfig:
    vmax_mm = raw.get("vmax_mm_s", {})
    vmax_deg = raw.get("vmax_deg_s", {})
    return SafetyConfig(
        min_confidence=float(raw.get("min_confidence", 0.6)),
        on_tracking_loss=str(raw.get("on_tracking_loss", "zero_offsets")),  # type: ignore[arg-type]
        on_low_confidence=str(raw.get("on_low_confidence", "zero_offsets")),  # type: ignore[arg-type]
        max_head_speed_mm_s=float(raw.get("max_head_speed_mm_s", 80.0)),
        vmax_mm_s=(
            float(vmax_mm.get("x", 30.0)),
            float(vmax_mm.get("y", 30.0)),
            float(vmax_mm.get("z", 15.0)),
        ),
        vmax_deg_s=(
            float(vmax_deg.get("b", 10.0)),
            float(vmax_deg.get("c", 10.0)),
        ),
        require_baseline_before_stream=bool(raw.get("require_baseline_before_stream", True)),
        min_standoff_mm=(
            float(raw["min_standoff_mm"]) if raw.get("min_standoff_mm") is not None else None
        ),
    )


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
        mm_to_axis_unit=float(data.get("mm_to_axis_unit", 1.0)),
        deg_to_axis_unit=float(data.get("deg_to_axis_unit", 1.0)),
        solver=str(data.get("solver", "decoupled")),  # type: ignore[arg-type]
        offset_mode=str(data.get("offset_mode", "follow")),  # type: ignore[arg-type]
        machine_pose=machine_pose,
        reference_normal=ref_normal,
        update_period_ms=float(data.get("update_period_ms", 10.0)),
        motor_map=parse_motor_axis_map(data.get("motor_map", "dual_x")),
    )
