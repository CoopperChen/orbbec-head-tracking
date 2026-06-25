from __future__ import annotations

import argparse
import contextlib
import os
import time
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from .cnc_config import CncCompensationConfig, MachinePose, load_compensation_config
from .cnc_offset_encoder import CncOffsetEncoder, CncUserOffset
from .cnc_protocol import UserOffsetMessage
from .cnc_safety import CncSafetyGuards
from .cnc_udp_streamer import CncUdpStreamer, CncUdpStreamerConfig
from .cnc_viz import draw_cnc_status_panel
from .config import TrackerConfig
from .tracker import OrbbecHeadTracker, colorize_depth_mm, draw_pose_overlay

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    args = _parse_args()
    comp_config = _build_compensation_config(args)
    tracker_config = TrackerConfig(
        pose_solver=args.pose_solver,
        smoothing_enabled=not args.no_smoothing,
        rotation_alpha=args.rotation_alpha if args.rotation_alpha is not None else TrackerConfig.rotation_alpha,
        rotation_deadband_deg=(
            args.rotation_deadband_deg
            if args.rotation_deadband_deg is not None
            else TrackerConfig.rotation_deadband_deg
        ),
        rotation_max_jump_deg=(
            args.rotation_max_jump_deg
            if args.rotation_max_jump_deg is not None
            else TrackerConfig.rotation_max_jump_deg
        ),
    )
    encoder = CncOffsetEncoder(comp_config)
    safety = CncSafetyGuards(comp_config)
    streamer = CncUdpStreamer(
        CncUdpStreamerConfig(
            device_ip=args.device_ip,
            bind_ip=args.bind_ip,
            device_port=args.device_port,
            update_period_ms=comp_config.update_period_ms,
            ack_timeout_ms=args.ack_timeout_ms,
            ack_watchdog_enabled=not args.no_ack_watchdog,
            motor_map=comp_config.motor_map,
        )
    )

    period_sec = comp_config.update_period_ms / 1000.0
    baseline_deadline = time.monotonic() + max(0.0, args.capture_baseline_sec)
    baseline_translations: list[np.ndarray] = []
    baseline_rvecs: list[np.ndarray] = []
    window_name = args.window_name
    depth_window_name = f"{window_name} Depth"

    with _suppress_native_stderr(not args.verbose):
        with OrbbecHeadTracker(tracker_config) as tracker:
            if args.view:
                if tracker.camera_matrix is None or tracker.distortion_coefficients is None:
                    raise RuntimeError("Tracker calibration is unavailable for --view")
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
            streamer.connect()
            next_tick = time.monotonic()
            try:
                while True:
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(min(0.001, next_tick - now))
                        continue
                    next_tick += period_sec

                    frame = tracker.read_frame()
                    tracking_ok = frame is not None and frame.pose is not None
                    confidence = 0.0
                    head_speed = 0.0
                    proposed = CncUserOffset.zero()
                    baseline_capturing = (
                        not encoder.baseline_ready and now <= baseline_deadline and tracking_ok
                    )

                    if tracking_ok and frame is not None and frame.pose is not None:
                        pose = frame.pose
                        confidence = float(getattr(pose, "confidence", 1.0))

                        if not encoder.baseline_ready:
                            if now <= baseline_deadline:
                                baseline_translations.append(
                                    np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
                                )
                                baseline_rvecs.append(
                                    np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3)
                                )
                            else:
                                _lock_baseline(
                                    encoder,
                                    pose,
                                    baseline_translations,
                                    baseline_rvecs,
                                )

                        if encoder.baseline_ready:
                            machine_pose = _resolve_machine_pose(args, comp_config)
                            proposed = encoder.encode(pose, machine_pose=machine_pose)
                            head_speed = safety.estimate_head_speed_mm_s(
                                np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3),
                                time.time(),
                            )

                    decision = safety.evaluate(
                        proposed,
                        tracking_ok=tracking_ok,
                        confidence=confidence,
                        baseline_ready=encoder.baseline_ready,
                        head_speed_mm_s=head_speed,
                        link_ok=streamer.link_ok,
                    )
                    message = UserOffsetMessage.from_xyzbc(
                        *decision.offset.as_tuple(),
                        motor_map=comp_config.motor_map,
                    )
                    streamer.send_message(message)

                    if args.verbose and decision.reason:
                        print(f"[safety] {decision.action}: {decision.reason}")

                    if args.view and frame is not None:
                        visualization = draw_pose_overlay(
                            frame,
                            tracker.camera_matrix,
                            tracker.distortion_coefficients,
                        )
                        visualization = draw_cnc_status_panel(
                            visualization,
                            baseline_ready=encoder.baseline_ready,
                            baseline_capturing=baseline_capturing,
                            offset=decision.offset,
                            decision=decision,
                            link_ok=streamer.link_ok,
                            link_label=streamer.link_status.label,
                            confidence=confidence,
                        )
                        cv2.imshow(window_name, visualization)
                        cv2.imshow(depth_window_name, colorize_depth_mm(frame.depth_mm))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break

            except KeyboardInterrupt:
                streamer.send_message(
                    UserOffsetMessage.from_xyzbc(
                        0,
                        0,
                        0,
                        0,
                        0,
                        motor_map=comp_config.motor_map,
                    )
                )
            finally:
                streamer.close()
                cv2.destroyAllWindows()


def _lock_baseline(
    encoder: CncOffsetEncoder,
    pose: Any,
    translations: list[np.ndarray],
    rvecs: list[np.ndarray],
) -> None:
    if not translations:
        encoder.capture_baseline(pose)
        return
    mean_t = np.mean(np.stack(translations, axis=0), axis=0).astype(np.float32)
    mean_r = np.mean(np.stack(rvecs, axis=0), axis=0).astype(np.float32)
    averaged = replace(
        pose,
        translation_vector_mm=mean_t,
        rotation_vector=mean_r.reshape(3, 1) if mean_r.ndim == 1 else mean_r,
    )
    encoder.capture_baseline(averaged)
    translations.clear()
    rvecs.clear()


def _resolve_machine_pose(args: argparse.Namespace, config: CncCompensationConfig) -> MachinePose | None:
    if args.machine_pose:
        return MachinePose.from_sequence([float(v) for v in args.machine_pose.split(",")])
    return config.machine_pose


def _build_compensation_config(args: argparse.Namespace) -> CncCompensationConfig:
    config = load_compensation_config(args.calibration) if args.calibration else CncCompensationConfig()
    return CncCompensationConfig(
        machine=config.machine,
        camera_to_machine_rotation=config.camera_to_machine_rotation,
        axis_limits=config.axis_limits,
        safety=config.safety,
        mm_to_axis_unit=config.mm_to_axis_unit,
        deg_to_axis_unit=config.deg_to_axis_unit,
        solver=args.solver or config.solver,
        offset_mode=args.offset_mode or config.offset_mode,
        machine_pose=config.machine_pose,
        reference_normal=config.reference_normal,
        update_period_ms=float(args.update_period_ms if args.update_period_ms is not None else config.update_period_ms),
        motor_map=config.motor_map,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Orbbec head pose as HICON XYZBC user offsets over UDP."
    )
    parser.add_argument("--device-ip", type=str, default="192.168.208.35")
    parser.add_argument("--bind-ip", type=str, default="192.168.208.10")
    parser.add_argument("--device-port", type=int, default=62095)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--machine-pose", type=str, default=None, help="X,Y,Z,B,C")
    parser.add_argument("--capture-baseline-sec", type=float, default=2.0)
    parser.add_argument("--update-period-ms", type=float, default=None)
    parser.add_argument("--ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--no-ack-watchdog",
        action="store_true",
        help="Do not fault when the controller sends no UDP ACK (view/dry-run only)",
    )
    parser.add_argument("--solver", choices=("kinematic", "decoupled"), default=None)
    parser.add_argument("--offset-mode", choices=("follow", "counter"), default=None)
    parser.add_argument("--pose-solver", choices=("depth-rigid", "pnp", "hybrid"), default="depth-rigid")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--rotation-alpha", type=float, default=None, help="Rotation SLERP factor 0-1; lower is steadier")
    parser.add_argument("--rotation-deadband-deg", type=float, default=None, help="Ignore rotation changes below this (deg)")
    parser.add_argument("--rotation-max-jump-deg", type=float, default=None, help="Reject raw rotation jumps above this (deg)")
    parser.add_argument("--view", action="store_true", help="Show live RGB pose + CNC offset panel")
    parser.add_argument("--window-name", type=str, default="Orbbec CNC Stream")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


@contextlib.contextmanager
def _suppress_native_stderr(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


if __name__ == "__main__":
    main()
