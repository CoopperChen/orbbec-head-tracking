from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from orbbec_head_tracking.cnc_config import MachinePose
from orbbec_head_tracking.cnc_work_pose_client import (
    WorkPoseUdpClient,
    parse_work_pose_payload,
)


def test_parse_work_pose_payload_mm() -> None:
    sample = parse_work_pose_payload(
        b'{"coord":"work","units":"mm","x":1.5,"y":2.5,"z":-3.5,"b":4.5,"c":5.5}'
    )
    assert sample.coordinate_system == "work"
    assert sample.pose == MachinePose(1.5, 2.5, -3.5, 4.5, 5.5)


def test_parse_work_pose_payload_inches() -> None:
    sample = parse_work_pose_payload(
        '{"coord":"work","units":"in","x":1.0,"y":0.0,"z":0.0,"b":10.0,"c":20.0}'
    )
    assert sample.pose.x == pytest.approx(25.4)
    assert sample.pose.b_deg == pytest.approx(10.0)


def test_parse_work_pose_rejects_machine_coords() -> None:
    with pytest.raises(ValueError, match="unsupported coordinate system"):
        parse_work_pose_payload('{"coord":"machine","x":0,"y":0,"z":0,"b":0,"c":0}')


def test_work_pose_udp_client_receives_latest() -> None:
    receiver = WorkPoseUdpClient(bind_ip="127.0.0.1", port=0, stale_sec=1.0)
    receiver.start()
    port = receiver._sock.getsockname()[1]  # type: ignore[union-attr]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        payload = json.dumps(
            {"coord": "work", "units": "mm", "x": 9.0, "y": 8.0, "z": 7.0, "b": 6.0, "c": 5.0}
        ).encode("utf-8")
        deadline = time.monotonic() + 2.0
        while receiver.latest() is None and time.monotonic() < deadline:
            sender.sendto(payload, ("127.0.0.1", port))
            time.sleep(0.02)
        pose = receiver.latest()
        assert pose == MachinePose(9.0, 8.0, 7.0, 6.0, 5.0)
        assert receiver.status_label().startswith("work pose: live")
    finally:
        sender.close()
        receiver.close()
