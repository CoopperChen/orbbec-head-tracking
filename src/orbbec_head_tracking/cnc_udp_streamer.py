from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from .cnc_protocol import (
    HICON_UDP_PORT,
    MotorAxisMap,
    UserOffsetMessage,
    is_set_axis_useroffset_ack,
    zero_xyzbc_message,
)


@dataclass(frozen=True)
class CncUdpStreamerConfig:
    device_ip: str
    bind_ip: str = "192.168.208.10"
    device_port: int = HICON_UDP_PORT
    update_period_ms: float = 10.0
    ack_timeout_ms: float = 2000.0
    ack_watchdog_enabled: bool = True
    send_zero_on_link_fault: bool = True
    motor_map: MotorAxisMap | None = None


@dataclass(frozen=True)
class LinkStatus:
    ok: bool
    label: str


class CncUdpStreamer:
    def __init__(self, config: CncUdpStreamerConfig) -> None:
        self.config = config
        self._sock: socket.socket | None = None
        self._device_addr: tuple[str, int] | None = None
        self._timeout_counter_ms: float = 0.0
        self._link_ok: bool = False
        self._link_label: str = "disconnected"
        self._receive_buffer = bytearray(1500)

    @property
    def link_ok(self) -> bool:
        return self._link_ok

    @property
    def link_status(self) -> LinkStatus:
        return LinkStatus(ok=self._link_ok, label=self._link_label)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        self._device_addr = None
        self._link_ok = False
        self._link_label = "disconnected"
        self._timeout_counter_ms = 0.0

    def connect(self) -> None:
        self.close()
        bind_ip = self.config.bind_ip
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((bind_ip, 0))
        sock.setblocking(False)
        self._sock = sock
        self._device_addr = (self.config.device_ip, int(self.config.device_port))
        self._timeout_counter_ms = 0.0
        self._link_ok = True
        self._link_label = (
            "waiting for ACK"
            if self.config.ack_watchdog_enabled
            else "open (ACK watchdog off)"
        )

    def _poll_ack(self) -> None:
        if self._sock is None:
            return
        while True:
            try:
                count, _addr = self._sock.recvfrom_into(self._receive_buffer)
            except BlockingIOError:
                break
            except OSError:
                self._link_ok = False
                self._link_label = "recv error"
                break
            if count > 0 and is_set_axis_useroffset_ack(bytes(self._receive_buffer[:count])):
                self._timeout_counter_ms = 0.0
                self._link_ok = True
                self._link_label = "ACK OK"
                break

    def tick_period(self) -> None:
        if not self.config.ack_watchdog_enabled:
            return
        self._timeout_counter_ms += float(self.config.update_period_ms)
        if self._timeout_counter_ms > float(self.config.ack_timeout_ms):
            self._link_ok = False
            self._link_label = "no ACK (timeout)"

    def send_message(self, message: UserOffsetMessage) -> bool:
        if self._sock is None or self._device_addr is None:
            return False
        self._poll_ack()
        self.tick_period()
        if not self._link_ok and self.config.send_zero_on_link_fault:
            message = zero_xyzbc_message(self.config.motor_map)
        try:
            self._sock.sendto(message.pack(), self._device_addr)
            return True
        except OSError:
            self._link_ok = False
            self._link_label = "send error"
            return False

    def maintain_loop_step(self, message: UserOffsetMessage) -> bool:
        if self._sock is None:
            self.connect()
        sent = self.send_message(message)
        if not sent:
            time.sleep(max(0.0, self.config.update_period_ms / 1000.0))
        return sent and self._link_ok
