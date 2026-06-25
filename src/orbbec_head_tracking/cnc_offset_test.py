from __future__ import annotations

import argparse
import time

from pathlib import Path

import cv2
import numpy as np

from .cnc_config import load_compensation_config
from .cnc_protocol import MotorAxisMap, UserOffsetMessage, parse_motor_axis_map, zero_xyzbc_message
from .cnc_udp_streamer import CncUdpStreamer, CncUdpStreamerConfig

_AXIS_ORDER = ("x", "y", "z", "b", "c")
_AXIS_LABELS = ("X mm", "Y mm", "Z mm", "B deg", "C deg")
_DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
    "x": (-10.0, 10.0),
    "y": (-10.0, 10.0),
    "z": (-10.0, 10.0),
    "b": (-30.0, 30.0),
    "c": (-30.0, 30.0),
}
_TRACKBAR_SCALE = 100


def _trackbar_max(min_value: float, max_value: float) -> int:
    span = max_value - min_value
    return max(1, int(round(span * _TRACKBAR_SCALE)))


def _trackbar_to_value(position: int, min_value: float, max_value: float) -> float:
    return min_value + (position / _TRACKBAR_SCALE)


def _value_to_trackbar(value: float, min_value: float, max_value: float) -> int:
    position = int(round((value - min_value) * _TRACKBAR_SCALE))
    return max(0, min(_trackbar_max(min_value, max_value), position))


def _parse_limits(spec: str) -> tuple[float, float]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected min,max got {spec!r}")
    return float(parts[0]), float(parts[1])


def _draw_panel(
    canvas: np.ndarray,
    *,
    connected: bool,
    manual_mode: bool,
    link_label: str,
    values: dict[str, float],
    device_ip: str,
    bind_ip: str,
    motor_map: MotorAxisMap,
) -> np.ndarray:
    out = canvas.copy()
    status_color = (70, 210, 110) if connected else (80, 90, 240)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), status_color, 2)
    rows = [
        "HICON XYZBC offset test",
        f"device {device_ip}:62095   bind {bind_ip}",
        f"{'CONNECTED' if connected else 'DISCONNECTED'}   link {link_label}",
        f"motors X{list(motor_map.x_motors)} Y{list(motor_map.y_motors)} Z{list(motor_map.z_motors)}",
        f"motors B{list(motor_map.b_motors)} C{list(motor_map.c_motors)}",
        f"mode {'MANUAL (paused stream)' if manual_mode else 'AUTO 100 Hz'}",
        (
            f"X {values['x']:7.2f}  Y {values['y']:7.2f}  Z {values['z']:7.2f}"
        ),
        (
            f"B {values['b']:7.2f}  C {values['c']:7.2f}"
        ),
        "c connect  d disconnect  m manual  s send  z zero  q quit",
    ]
    for index, text in enumerate(rows):
        cv2.putText(
            out,
            text,
            (16, 34 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 244, 252),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def _read_trackbar_values(
    window_name: str,
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for axis in _AXIS_ORDER:
        minimum, maximum = limits[axis]
        position = cv2.getTrackbarPos(axis, window_name)
        values[axis] = _trackbar_to_value(position, minimum, maximum)
    return values


def _set_trackbar_values(
    window_name: str,
    limits: dict[str, tuple[float, float]],
    values: dict[str, float],
) -> None:
    for axis in _AXIS_ORDER:
        minimum, maximum = limits[axis]
        cv2.setTrackbarPos(
            axis,
            window_name,
            _value_to_trackbar(values[axis], minimum, maximum),
        )


def _resolve_motor_map(spec: str) -> MotorAxisMap:
    path = Path(spec)
    if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
        return load_compensation_config(path).motor_map
    return parse_motor_axis_map(spec)


def run_panel(args: argparse.Namespace) -> int:
    limits = dict(_DEFAULT_LIMITS)
    if args.xyz_limits is not None:
        xyz_min, xyz_max = _parse_limits(args.xyz_limits)
        limits["x"] = (xyz_min, xyz_max)
        limits["y"] = (xyz_min, xyz_max)
        limits["z"] = (xyz_min, xyz_max)
    if args.bc_limits is not None:
        bc_min, bc_max = _parse_limits(args.bc_limits)
        limits["b"] = (bc_min, bc_max)
        limits["c"] = (bc_min, bc_max)

    window_name = "CNC XYZBC Offset Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 760, 360)

    motor_map = _resolve_motor_map(args.motor_map)

    for axis, label in zip(_AXIS_ORDER, _AXIS_LABELS, strict=True):
        minimum, maximum = limits[axis]
        cv2.createTrackbar(
            axis,
            window_name,
            _value_to_trackbar(0.0, minimum, maximum),
            _trackbar_max(minimum, maximum),
            lambda _value, axis_name=axis, lim=limits[axis]: None,
        )
        cv2.setTrackbarMin(axis, window_name, 0)
        cv2.setTrackbarMax(axis, window_name, _trackbar_max(minimum, maximum))

    streamer = CncUdpStreamer(
        CncUdpStreamerConfig(
            device_ip=args.device_ip,
            bind_ip=args.bind_ip,
            update_period_ms=args.update_period_ms,
            ack_timeout_ms=args.ack_timeout_ms,
            ack_watchdog_enabled=not args.no_ack_watchdog,
            send_zero_on_link_fault=False,
            motor_map=motor_map,
        )
    )
    connected = False
    manual_mode = False
    period_sec = max(0.001, args.update_period_ms / 1000.0)

    if args.connect:
        streamer.connect()
        connected = True

    try:
        while True:
            values = _read_trackbar_values(window_name, limits)
            if connected and not manual_mode:
                message = UserOffsetMessage.from_xyzbc(
                    values["x"],
                    values["y"],
                    values["z"],
                    values["b"],
                    values["c"],
                    motor_map=motor_map,
                )
                streamer.send_message(message)
                if args.no_ack_watchdog:
                    pass
                elif not streamer.link_ok:
                    connected = False
                    streamer.close()

            panel = _draw_panel(
                np.zeros((340, 760, 3), dtype=np.uint8),
                connected=connected,
                manual_mode=manual_mode,
                link_label=streamer.link_status.label,
                values=values,
                device_ip=args.device_ip,
                bind_ip=args.bind_ip,
                motor_map=motor_map,
            )
            cv2.imshow(window_name, panel)

            key = cv2.waitKey(int(round(period_sec * 1000.0))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and not connected:
                streamer.connect()
                connected = True
            elif key == ord("d") and connected:
                streamer.send_message(zero_xyzbc_message(motor_map))
                streamer.close()
                connected = False
            elif key == ord("m"):
                manual_mode = not manual_mode
            elif key == ord("s") and connected:
                streamer.send_message(
                    UserOffsetMessage.from_xyzbc(
                        values["x"],
                        values["y"],
                        values["z"],
                        values["b"],
                        values["c"],
                        motor_map=motor_map,
                    )
                )
            elif key == ord("z"):
                zero_values = {axis: 0.0 for axis in _AXIS_ORDER}
                _set_trackbar_values(window_name, limits, zero_values)
                if connected:
                    streamer.send_message(zero_xyzbc_message(motor_map))
    finally:
        if connected:
            streamer.send_message(zero_xyzbc_message(motor_map))
        streamer.close()
        cv2.destroyAllWindows()
    return 0


def run_once(args: argparse.Namespace) -> int:
    values = [float(part.strip()) for part in args.offsets.split(",")]
    if len(values) != 5:
        raise ValueError("--offsets requires 5 comma-separated values: X,Y,Z,B,C")

    motor_map = _resolve_motor_map(args.motor_map)
    streamer = CncUdpStreamer(
        CncUdpStreamerConfig(
            device_ip=args.device_ip,
            bind_ip=args.bind_ip,
            update_period_ms=args.update_period_ms,
            ack_timeout_ms=args.ack_timeout_ms,
            ack_watchdog_enabled=not args.no_ack_watchdog,
            send_zero_on_link_fault=False,
            motor_map=motor_map,
        )
    )
    streamer.connect()
    message = UserOffsetMessage.from_xyzbc(*values, motor_map=motor_map)
    deadline = time.monotonic() + max(0.0, args.duration_sec)
    try:
        while time.monotonic() < deadline:
            streamer.send_message(message)
            time.sleep(max(0.001, args.update_period_ms / 1000.0))
            if not args.no_ack_watchdog and streamer.link_ok and streamer.link_status.label == "ACK OK":
                print("ACK received")
                break
        print(
            f"sent XYZBC={tuple(values)} link={streamer.link_status.label} "
            f"ok={streamer.link_ok}"
        )
    finally:
        streamer.send_message(zero_xyzbc_message(motor_map))
        streamer.close()
    return 0 if streamer.link_ok or args.no_ack_watchdog else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manual HICON XYZBC user-offset test. "
            "OpenCV trackbars mirror MotionUserOffsetSample sliders."
        )
    )
    parser.add_argument("--device-ip", default="192.168.208.35")
    parser.add_argument("--bind-ip", default="192.168.208.10")
    parser.add_argument("--update-period-ms", type=float, default=10.0)
    parser.add_argument("--ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--no-ack-watchdog",
        action="store_true",
        help="Keep streaming without faulting when no controller ACK arrives",
    )
    parser.add_argument(
        "--xyz-limits",
        help="Min,max mm for X/Y/Z trackbars (default -10,10)",
    )
    parser.add_argument(
        "--bc-limits",
        help="Min,max deg for B/C trackbars (default -30,30)",
    )
    parser.add_argument(
        "--motor-map",
        default="dual_x",
        help="Motor map preset (dual_x, standard) or calibration YAML path",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Connect to the controller immediately when the panel opens",
    )
    parser.add_argument(
        "--offsets",
        help="Send fixed X,Y,Z,B,C once or for --duration-sec without opening the panel",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=2.0,
        help="How long to stream fixed --offsets before zeroing (default 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offsets is not None:
        return run_once(args)
    return run_panel(args)


if __name__ == "__main__":
    raise SystemExit(main())
