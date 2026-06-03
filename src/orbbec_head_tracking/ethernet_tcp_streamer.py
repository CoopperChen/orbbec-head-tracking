from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TcpJsonlStreamerConfig:
    host: str
    port: int
    reconnect_interval_sec: float = 1.0
    connect_timeout_sec: float = 3.0


class TcpJsonlStreamer:
    def __init__(self, config: TcpJsonlStreamerConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.config.connect_timeout_sec)
        sock.connect((self.config.host, int(self.config.port)))
        sock.settimeout(None)
        self._sock = sock

    def _ensure_connected(self) -> None:
        while self._sock is None:
            try:
                self._connect()
            except OSError:
                time.sleep(max(0.0, float(self.config.reconnect_interval_sec)))

    def send_json_obj(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        while True:
            self._ensure_connected()
            assert self._sock is not None
            try:
                self._sock.sendall(data)
                return
            except OSError:
                self.close()
                time.sleep(max(0.0, float(self.config.reconnect_interval_sec)))
