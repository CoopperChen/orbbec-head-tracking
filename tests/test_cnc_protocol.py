from __future__ import annotations

import struct

import numpy as np
import pytest

from orbbec_head_tracking.cnc_protocol import (
    AXIS_B,
    AXIS_C,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    MESSAGE_SIGN,
    MSG_SET_AXIS_USEROFFSET,
    MotorAxisMap,
    REPLY_TYPE_SET_AXIS_USEROFFSET,
    UserOffsetMessage,
    is_set_axis_useroffset_ack,
    message_size,
)


def test_message_size_is_48_bytes() -> None:
    assert message_size() == 48


def test_pack_xyzbc_layout() -> None:
    msg = UserOffsetMessage.from_xyzbc(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        motor_map=MotorAxisMap.standard(),
    )
    data = msg.pack()
    assert len(data) == 48
    sign, msg_type = struct.unpack("<II", data[:8])
    assert sign == MESSAGE_SIGN
    assert msg_type == MSG_SET_AXIS_USEROFFSET
    enables = struct.unpack("<8B", data[8:16])
    assert enables[:5] == (1, 1, 1, 1, 1)
    assert enables[5:] == (0, 0, 0)
    offsets = struct.unpack("<8f", data[16:48])
    assert offsets[0] == pytest.approx(1.0)
    assert offsets[1] == pytest.approx(2.0)
    assert offsets[2] == pytest.approx(3.0)
    assert offsets[3] == pytest.approx(4.0)
    assert offsets[4] == pytest.approx(5.0)


def test_roundtrip_unpack() -> None:
    original = UserOffsetMessage.from_xyzbc(
        -0.5,
        0.25,
        1.5,
        -2.0,
        3.0,
        motor_map=MotorAxisMap.standard(),
    )
    restored = UserOffsetMessage.unpack(original.pack())
    assert restored.axis_offsets[AXIS_X] == pytest.approx(-0.5)
    assert restored.axis_offsets[AXIS_Y] == pytest.approx(0.25)
    assert restored.axis_offsets[AXIS_Z] == pytest.approx(1.5)
    assert restored.axis_offsets[AXIS_B] == pytest.approx(-2.0)
    assert restored.axis_offsets[AXIS_C] == pytest.approx(3.0)


def test_ack_detection() -> None:
    ack = struct.pack("<II", MESSAGE_SIGN, REPLY_TYPE_SET_AXIS_USEROFFSET)
    assert is_set_axis_useroffset_ack(ack)
    assert not is_set_axis_useroffset_ack(struct.pack("<II", MESSAGE_SIGN, MSG_SET_AXIS_USEROFFSET))


def test_pack_dual_x_motor_map() -> None:
    msg = UserOffsetMessage.from_xyzbc(1.0, 2.0, 3.0, 4.0, 5.0, motor_map=MotorAxisMap.dual_x())
    data = msg.pack()
    enables = struct.unpack("<8B", data[8:16])
    offsets = struct.unpack("<8f", data[16:48])
    assert enables[0] == 1
    assert enables[3] == 1
    assert offsets[0] == pytest.approx(1.0)
    assert offsets[3] == pytest.approx(1.0)
    assert enables[1] == 1
    assert offsets[1] == pytest.approx(2.0)
    assert enables[2] == 1
    assert offsets[2] == pytest.approx(3.0)
    assert enables[4] == 1
    assert offsets[4] == pytest.approx(4.0)
    assert enables[5] == 1
    assert offsets[5] == pytest.approx(5.0)
    assert enables[6:] == (0, 0)
