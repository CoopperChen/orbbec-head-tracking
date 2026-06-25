from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Sequence

MESSAGE_SIGN = 0xAA4321BB
BASE_MSG_TYPE = 0xB0B0
MSG_SET_AXIS_USEROFFSET = BASE_MSG_TYPE + 300
REPLY_MSG_TYPE = 0x40000000
REPLY_TYPE_SET_AXIS_USEROFFSET = REPLY_MSG_TYPE | MSG_SET_AXIS_USEROFFSET
HICON_UDP_PORT = 62095

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
AXIS_B = 3
AXIS_C = 4
AXIS_COUNT = 8
XYZBC_AXES = (AXIS_X, AXIS_Y, AXIS_Z, AXIS_B, AXIS_C)
LOGICAL_AXIS_NAMES = ("x", "y", "z", "b", "c")

_PACK_FMT = "<II8B8f"
_MESSAGE_SIZE = struct.calcsize(_PACK_FMT)


@dataclass(frozen=True)
class MotorAxisMap:
    """Maps logical XYZBC compensation to HICON motor indices (0-7)."""

    x_motors: tuple[int, ...] = (AXIS_X,)
    y_motors: tuple[int, ...] = (AXIS_Y,)
    z_motors: tuple[int, ...] = (AXIS_Z,)
    b_motors: tuple[int, ...] = (AXIS_B,)
    c_motors: tuple[int, ...] = (AXIS_C,)

    @classmethod
    def standard(cls) -> MotorAxisMap:
        return cls()

    @classmethod
    def dual_x(cls) -> MotorAxisMap:
        """X gantry: motor0 + motor3 share the same X offset; B/C use motors 4/5."""
        return cls(
            x_motors=(0, 3),
            y_motors=(1,),
            z_motors=(2,),
            b_motors=(4,),
            c_motors=(5,),
        )

    def assignments(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return (
            ("x", self.x_motors),
            ("y", self.y_motors),
            ("z", self.z_motors),
            ("b", self.b_motors),
            ("c", self.c_motors),
        )

    def validate(self) -> None:
        seen: dict[int, str] = {}
        for axis_name, motors in self.assignments():
            if not motors:
                raise ValueError(f"motor_map.{axis_name} must list at least one motor index")
            for motor_idx in motors:
                if motor_idx < 0 or motor_idx >= AXIS_COUNT:
                    raise ValueError(
                        f"motor_map.{axis_name} motor {motor_idx} out of range 0..{AXIS_COUNT - 1}"
                    )
                if motor_idx in seen:
                    raise ValueError(
                        f"motor {motor_idx} assigned to both {seen[motor_idx]} and {axis_name}"
                    )
                seen[motor_idx] = axis_name


def _normalize_motor_list(raw: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, int):
        return (raw,)
    motors = tuple(int(value) for value in raw)
    if not motors:
        raise ValueError("motor list must not be empty")
    return motors


def parse_motor_axis_map(raw: object | None) -> MotorAxisMap:
    if raw is None:
        return MotorAxisMap.dual_x()
    if isinstance(raw, str):
        preset = raw.strip().lower()
        if preset in ("dual_x", "dual-x", "dualx"):
            return MotorAxisMap.dual_x()
        if preset in ("standard", "default", "1to1"):
            return MotorAxisMap.standard()
        raise ValueError(f"unknown motor_map preset: {raw!r}")
    if not isinstance(raw, dict):
        raise ValueError("motor_map must be a preset string or mapping")

    defaults = MotorAxisMap.dual_x()
    motor_map = MotorAxisMap(
        x_motors=_normalize_motor_list(raw.get("x", defaults.x_motors)),
        y_motors=_normalize_motor_list(raw.get("y", defaults.y_motors)),
        z_motors=_normalize_motor_list(raw.get("z", defaults.z_motors)),
        b_motors=_normalize_motor_list(raw.get("b", defaults.b_motors)),
        c_motors=_normalize_motor_list(raw.get("c", defaults.c_motors)),
    )
    motor_map.validate()
    return motor_map


@dataclass
class UserOffsetMessage:
    message_sign: int = MESSAGE_SIGN
    msg_type: int = MSG_SET_AXIS_USEROFFSET
    axis_offset_enable: list[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 1, 0, 0, 0]
    )
    axis_offsets: list[float] = field(default_factory=lambda: [0.0] * AXIS_COUNT)

    def __post_init__(self) -> None:
        if len(self.axis_offset_enable) != AXIS_COUNT:
            raise ValueError("axis_offset_enable must have length 8")
        if len(self.axis_offsets) != AXIS_COUNT:
            raise ValueError("axis_offsets must have length 8")

    @classmethod
    def from_xyzbc(
        cls,
        x: float,
        y: float,
        z: float,
        b: float,
        c: float,
        *,
        enable_xyzbc: bool = True,
        motor_map: MotorAxisMap | None = None,
    ) -> UserOffsetMessage:
        enable = [0] * AXIS_COUNT
        offsets = [0.0] * AXIS_COUNT
        axis_map = motor_map or MotorAxisMap.standard()
        axis_map.validate()
        values = (x, y, z, b, c)
        for (_axis_name, motors), value in zip(axis_map.assignments(), values, strict=True):
            for motor_idx in motors:
                enable[motor_idx] = 1 if enable_xyzbc else 0
                offsets[motor_idx] = float(value) if enable_xyzbc else 0.0
        return cls(axis_offset_enable=enable, axis_offsets=offsets)

    def pack(self) -> bytes:
        return struct.pack(
            _PACK_FMT,
            int(self.message_sign),
            int(self.msg_type),
            *[int(v) & 0xFF for v in self.axis_offset_enable],
            *[float(v) for v in self.axis_offsets],
        )

    @classmethod
    def unpack(cls, data: bytes) -> UserOffsetMessage:
        if len(data) < _MESSAGE_SIZE:
            raise ValueError(f"expected at least {_MESSAGE_SIZE} bytes, got {len(data)}")
        values = struct.unpack(_PACK_FMT, data[:_MESSAGE_SIZE])
        return cls(
            message_sign=int(values[0]),
            msg_type=int(values[1]),
            axis_offset_enable=[int(v) for v in values[2:10]],
            axis_offsets=[float(v) for v in values[10:18]],
        )


def unpack_reply_header(data: bytes) -> tuple[int, int] | None:
    if len(data) < 8:
        return None
    sign, msg_type = struct.unpack("<II", data[:8])
    return int(sign), int(msg_type)


def is_set_axis_useroffset_ack(data: bytes) -> bool:
    header = unpack_reply_header(data)
    if header is None:
        return False
    sign, msg_type = header
    return sign == MESSAGE_SIGN and msg_type == REPLY_TYPE_SET_AXIS_USEROFFSET


def message_size() -> int:
    return _MESSAGE_SIZE


def zero_xyzbc_message(motor_map: MotorAxisMap | None = None) -> UserOffsetMessage:
    return UserOffsetMessage.from_xyzbc(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        motor_map=motor_map,
    )


def offsets_tuple(message: UserOffsetMessage) -> tuple[float, float, float, float, float]:
    o = message.axis_offsets
    return float(o[AXIS_X]), float(o[AXIS_Y]), float(o[AXIS_Z]), float(o[AXIS_B]), float(o[AXIS_C])
