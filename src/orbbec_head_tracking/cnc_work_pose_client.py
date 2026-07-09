from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from .cnc_config import MachinePose


@dataclass(frozen=True)
class WorkPoseSample:
    pose: MachinePose
    received_at: float
    coordinate_system: str = "work"


class WorkPoseUdpClient:
    """Receive active work-coordinate XYZBC from Mach4 over UDP (JSON)."""

    def __init__(
        self,
        *,
        bind_ip: str = "0.0.0.0",
        port: int = 62100,
        stale_sec: float = 0.5,
    ) -> None:
        if port < 0 or port > 65535:
            raise ValueError("work pose UDP port must be 0..65535")
        self.bind_ip = bind_ip
        self.port = int(port)
        self.stale_sec = max(0.0, float(stale_sec))
        self._lock = threading.Lock()
        self._latest: WorkPoseSample | None = None
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._packets = 0
        self._parse_errors = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_ip, self.port))
        sock.settimeout(0.25)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, name="work-pose-udp", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def packets(self) -> int:
        with self._lock:
            return self._packets

    @property
    def parse_errors(self) -> int:
        with self._lock:
            return self._parse_errors

    def latest(self, *, now: float | None = None) -> MachinePose | None:
        sample = self._latest_sample(now=now)
        return sample.pose if sample is not None else None

    def status_label(self, *, now: float | None = None) -> str:
        sample = self._latest_sample(now=now)
        if sample is None:
            with self._lock:
                if self._latest is None:
                    return "work pose: waiting"
                return "work pose: stale"
        age_ms = (time.monotonic() if now is None else now) - sample.received_at
        return f"work pose: live ({age_ms * 1000.0:.0f} ms)"

    def _latest_sample(self, *, now: float | None = None) -> WorkPoseSample | None:
        clock = time.monotonic() if now is None else now
        with self._lock:
            sample = self._latest
        if sample is None:
            return None
        if self.stale_sec > 0.0 and clock - sample.received_at > self.stale_sec:
            return None
        return sample

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue
            try:
                sample = parse_work_pose_payload(data)
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                with self._lock:
                    self._parse_errors += 1
                continue
            with self._lock:
                self._latest = sample
                self._packets += 1


def parse_work_pose_payload(data: bytes | str) -> WorkPoseSample:
    if isinstance(data, bytes):
        text = data.decode("utf-8").strip()
    else:
        text = data.strip()
    if not text:
        raise ValueError("empty work pose payload")
    record = json.loads(text)
    if not isinstance(record, dict):
        raise ValueError("work pose payload must be a JSON object")
    return WorkPoseSample(
        pose=_record_to_pose(record),
        received_at=time.monotonic(),
        coordinate_system=str(record.get("coord", record.get("coordinate_system", "work"))),
    )


def _record_to_pose(record: dict[str, Any]) -> MachinePose:
    coord = str(record.get("coord", record.get("coordinate_system", "work"))).lower()
    if coord not in ("work", "g54", "active"):
        raise ValueError(f"unsupported coordinate system {coord!r}; expected work")

    def value(*keys: str) -> float:
        for key in keys:
            if key in record:
                return float(record[key])
        raise KeyError(keys[0])

    inch_to_mm = 25.4
    units = str(record.get("units", "mm")).lower()
    scale = inch_to_mm if units in ("in", "inch", "inches") else 1.0

    return MachinePose(
        x=value("x", "X") * scale,
        y=value("y", "Y") * scale,
        z=value("z", "Z") * scale,
        b_deg=value("b", "B", "b_deg"),
        c_deg=value("c", "C", "c_deg"),
    )
