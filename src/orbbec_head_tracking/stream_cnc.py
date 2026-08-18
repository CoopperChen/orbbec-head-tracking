from __future__ import annotations

import argparse
import contextlib
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cnc_config import CncCompensationConfig, MachinePose, OffsetDeadbandConfig, load_compensation_config
from .cnc_mismatch import CncMismatchTracker
from .cnc_offset_encoder import CncOffsetEncoder, CncUserOffset, offset_is_zero
from .cnc_protocol import UserOffsetMessage
from .cnc_safety import CncSafetyGuards, SafetyDecision
from .cnc_stability_log import StabilityCsvLogger
from .cnc_udp_streamer import CncUdpStreamer, CncUdpStreamerConfig
from .cnc_viz import draw_cnc_status_panel
from .cnc_work_pose_client import WorkPoseUdpClient
from .config import TrackerConfig
from .tracker import OrbbecHeadTracker, colorize_depth_mm, draw_pose_overlay

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_LOG_DIR = Path("results")
DEFAULT_LOG_STEM = "cnc_stream"


def main() -> None:
    args = _parse_args()
    comp_config = _build_compensation_config(args)
    tracker_config = _build_tracker_config(args)
    encoder = CncOffsetEncoder(comp_config)
    mismatch = CncMismatchTracker(comp_config)
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
    work_pose_client = _open_work_pose_client(args, comp_config)
    logger = _open_stability_logger(args)

    period_sec = comp_config.update_period_ms / 1000.0
    duration_sec = args.duration_sec if args.duration_sec and args.duration_sec > 0 else None
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
            # Time from the first loop tick, so camera start-up does not eat the
            # baseline window or the requested run duration.
            start_time = time.monotonic()
            baseline_deadline = start_time + max(0.0, args.capture_baseline_sec)
            next_tick = start_time
            last_sent = CncUserOffset.zero()
            prev_tick_sec: float | None = None
            step_monitor = StepMonitor(args.step_warn_mm, args.step_warn_deg)
            try:
                while True:
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(min(0.001, next_tick - now))
                        continue
                    next_tick += period_sec
                    elapsed = now - start_time
                    loop_dt_ms = (
                        (now - prev_tick_sec) * 1000.0 if prev_tick_sec is not None else None
                    )
                    prev_tick_sec = now

                    frame = tracker.read_frame()
                    tracking_ok = frame is not None and frame.pose is not None
                    pose = frame.pose if frame is not None else None
                    confidence = 0.0
                    head_speed = 0.0
                    proposed = CncUserOffset.zero()
                    target = CncUserOffset.zero()
                    head_delta = None
                    mismatch_report = None
                    baseline_capturing = (
                        not encoder.baseline_ready and now <= baseline_deadline and tracking_ok
                    )

                    require_live_work_pose = args.require_work_pose or comp_config.work_pose_udp.require_live
                    work_pose_missing = False
                    if tracking_ok and pose is not None:
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
                                    mismatch,
                                    _resolve_tool_pose(args, comp_config, work_pose_client),
                                    pose,
                                    baseline_translations,
                                    baseline_rvecs,
                                )
                                safety.reset()

                        if encoder.baseline_ready:
                            tool_pose = _resolve_tool_pose(args, comp_config, work_pose_client)
                            if tool_pose is not None:
                                proposed = encoder.encode(pose, machine_pose=tool_pose)
                                head_speed = safety.estimate_head_speed_mm_s(
                                    np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3),
                                    now,
                                )
                                head_delta = encoder.head_delta_machine_mm(pose)
                                target, mismatch_report = mismatch.target(
                                    proposed,
                                    safety.last_offset,
                                    dt_sec=period_sec,
                                    head_speed_mm_s=head_speed,
                                    d_t_machine=head_delta,
                                    preserve_sent=safety.in_recovery,
                                )
                            else:
                                work_pose_missing = require_live_work_pose
                                if args.verbose:
                                    print("[work-pose] waiting for live work coordinates")
                        else:
                            target = safety.last_offset
                    else:
                        target = safety.last_offset

                    if work_pose_missing:
                        decision = SafetyDecision("hold_last", safety.last_offset, "work_pose_missing")
                    else:
                        decision = safety.evaluate(
                            target,
                            spike_reference=proposed if tracking_ok and encoder.baseline_ready else None,
                            tracking_ok=tracking_ok,
                            confidence=confidence,
                            baseline_ready=encoder.baseline_ready,
                            head_speed_mm_s=head_speed,
                            link_ok=streamer.link_ok,
                            now_sec=now,
                            snap=(
                                mismatch_report is not None and mismatch_report.mode == "snap"
                            ),
                        )
                    if decision.action == "hold_last":
                        mismatch.notify_command_hold(decision.reason)
                    message = UserOffsetMessage.from_xyzbc(
                        *decision.offset.as_tuple(),
                        motor_map=comp_config.motor_map,
                    )
                    streamer.send_message(message)
                    step_mm, step_deg = step_monitor.observe(last_sent, decision.offset, now)
                    last_sent = decision.offset

                    if logger is not None:
                        logger.record(
                            now_sec=now,
                            elapsed_sec=elapsed,
                            pose=pose if tracking_ok else None,
                            head_delta_machine_mm=head_delta,
                            required=proposed if encoder.baseline_ready else None,
                            sent=decision.offset,
                            confidence=confidence,
                            head_speed_mm_s=head_speed,
                            tracking_ok=tracking_ok,
                            baseline_ready=encoder.baseline_ready,
                            link_ok=streamer.link_ok,
                            safety_action=decision.action,
                            safety_reason=decision.reason,
                            loop_dt_ms=loop_dt_ms,
                            step_mm=step_mm,
                            step_deg=step_deg,
                        )

                    if args.verbose and decision.reason:
                        print(f"[safety] {decision.action}: {decision.reason}")

                    if args.view and frame is not None:
                        display_tool_pose = _resolve_tool_pose(
                            args, comp_config, work_pose_client
                        )
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
                            required=proposed if encoder.baseline_ready else None,
                            mismatch_report=mismatch_report,
                            decision=decision,
                            link_ok=streamer.link_ok,
                            link_label=streamer.link_status.label,
                            confidence=confidence,
                            work_pose_status=(
                                work_pose_client.status_label()
                                if work_pose_client is not None
                                else None
                            ),
                            tool_pose=display_tool_pose,
                        )
                        cv2.imshow(window_name, visualization)
                        cv2.imshow(depth_window_name, colorize_depth_mm(frame.depth_mm))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            break

                    if duration_sec is not None and elapsed >= duration_sec:
                        print(f"run complete after {elapsed / 60.0:.1f} min")
                        break

            except KeyboardInterrupt:
                pass
            finally:
                print(step_monitor.summary())
                _report_standing_offset(last_sent)
                streamer.close()
                if work_pose_client is not None:
                    work_pose_client.close()
                if logger is not None:
                    logger.close()
                    print(f"stability log: {logger.rows_written} rows -> {logger.path}")
                cv2.destroyAllWindows()


class StepMonitor:
    """Tracks how big each commanded offset change is.

    The controller applies a user offset as a position change inside one servo
    cycle, so a large single-packet step is a velocity transient at the drive
    however slowly the offset is moving on average.
    """

    def __init__(self, warn_mm: float, warn_deg: float, *, cooldown_sec: float = 1.0) -> None:
        self.warn_mm = float(warn_mm)
        self.warn_deg = float(warn_deg)
        self._cooldown_sec = float(cooldown_sec)
        self._next_warn_sec = 0.0
        self.max_mm = 0.0
        self.max_deg = 0.0
        self.over_threshold = 0

    def observe(
        self,
        previous: CncUserOffset,
        current: CncUserOffset,
        now_sec: float,
    ) -> tuple[float, float]:
        step_mm = float(
            np.linalg.norm(
                [current.x - previous.x, current.y - previous.y, current.z - previous.z]
            )
        )
        step_deg = float(np.linalg.norm([current.b - previous.b, current.c - previous.c]))
        self.max_mm = max(self.max_mm, step_mm)
        self.max_deg = max(self.max_deg, step_deg)

        exceeded = (self.warn_mm > 0.0 and step_mm > self.warn_mm) or (
            self.warn_deg > 0.0 and step_deg > self.warn_deg
        )
        if exceeded:
            self.over_threshold += 1
            if now_sec >= self._next_warn_sec:
                print(f"[step] {step_mm:.2f} mm / {step_deg:.2f} deg in one packet")
                self._next_warn_sec = now_sec + self._cooldown_sec
        return step_mm, step_deg

    def summary(self) -> str:
        return (
            f"max single-packet step: {self.max_mm:.2f} mm / {self.max_deg:.2f} deg"
            f" ({self.over_threshold} over threshold)"
        )


#: A standing offset above this is called out in a banner rather than one line.
EXIT_OFFSET_WARN_MM = 5.0


def _report_standing_offset(last_sent: CncUserOffset) -> None:
    """Report any offset left in the controller. Never commands motion.

    Zeroing on exit would move the tool by the full standing offset relative to
    the head, which is unacceptable with a probe near the scalp. So the offset
    is left alone and the operator clears it in Mach4 once the tool is clear.

    The catch is that the tracker cannot read the controller's current offset:
    it assumes zero at startup and sends a bare zero while waiting for the
    baseline. So an uncleared offset becomes a single-step jump on the next run,
    which is exactly why this warning has to be impossible to miss.
    """
    if offset_is_zero(last_sent):
        return

    x, y, z, b, c = last_sent.as_tuple()
    magnitude = float(np.linalg.norm((x, y, z)))
    values = f"X{x:+.3f} Y{y:+.3f} Z{z:+.3f} B{b:+.3f} C{c:+.3f}"

    if magnitude < EXIT_OFFSET_WARN_MM:
        print(f"standing offset left in controller ({magnitude:.2f} mm): {values}")
        print("clear it in Mach4 before the next run.")
        return

    bar = "!" * 74
    print(f"\n{bar}")
    print(f"!! STANDING OFFSET LEFT IN THE CONTROLLER: {magnitude:.2f} mm")
    print(f"!!   {values}")
    print("!!")
    print("!! The machine was NOT moved -- zeroing this would have dragged the tool")
    print("!! that far relative to the head.")
    print("!!")
    print("!! Clear it in Mach4 once the tool is clear of the workpiece.")
    print("!! Until you do, starting this tool again commands zero in a single step,")
    print(f"!! which the controller applies as a {magnitude:.2f} mm jump inside one servo cycle.")
    print(f"{bar}\n")


def _lock_baseline(
    encoder: CncOffsetEncoder,
    mismatch: CncMismatchTracker,
    machine_pose: MachinePose | None,
    pose: Any,
    translations: list[np.ndarray],
    rvecs: list[np.ndarray],
) -> None:
    if not translations:
        encoder.capture_baseline(pose, machine_pose=machine_pose)
    else:
        mean_t = np.mean(np.stack(translations, axis=0), axis=0).astype(np.float32)
        mean_r = np.mean(np.stack(rvecs, axis=0), axis=0).astype(np.float32)
        averaged = replace(
            pose,
            translation_vector_mm=mean_t,
            rotation_vector=mean_r.reshape(3, 1) if mean_r.ndim == 1 else mean_r,
        )
        encoder.capture_baseline(averaged, machine_pose=machine_pose)
        translations.clear()
        rvecs.clear()

    resolved_pose = machine_pose or MachinePose(0.0, 0.0, 0.0, 0.0, 0.0)
    mismatch.capture_baseline(resolved_pose)


def _resolve_tool_pose(
    args: argparse.Namespace,
    config: CncCompensationConfig,
    work_pose_client: WorkPoseUdpClient | None,
) -> MachinePose | None:
    if work_pose_client is not None:
        live = work_pose_client.latest()
        if live is not None:
            return live
        if args.require_work_pose or config.work_pose_udp.require_live:
            return None
    if args.machine_pose:
        return MachinePose.from_sequence([float(v) for v in args.machine_pose.split(",")])
    return config.machine_pose


def _open_stability_logger(args: argparse.Namespace) -> StabilityCsvLogger | None:
    path = _resolve_log_path(args)
    if path is None:
        return None
    logger = StabilityCsvLogger(path, rate_hz=args.log_rate_hz).open()
    print(f"stability log -> {path} at {args.log_rate_hz:g} Hz")
    return logger


def _resolve_log_path(args: argparse.Namespace) -> Path | None:
    """CSV destination for ``--log`` (always timestamped) or ``--log-csv``."""
    if args.log is not None:
        return _stamped(_log_target(args.log))
    if not args.log_csv:
        return None
    path = Path(args.log_csv)
    return _stamped(path) if args.log_timestamped else path


def _log_target(value: str) -> Path:
    """Interpret a ``--log`` value as a full CSV path or a directory to write into."""
    if not value:
        return DEFAULT_LOG_DIR / f"{DEFAULT_LOG_STEM}.csv"
    path = Path(value)
    if path.suffix and not path.is_dir():
        return path
    return path / f"{DEFAULT_LOG_STEM}.csv"


def _stamped(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix or '.csv'}")


def _open_work_pose_client(
    args: argparse.Namespace,
    config: CncCompensationConfig,
) -> WorkPoseUdpClient | None:
    port = args.work_pose_udp_port
    if port is None:
        port = config.work_pose_udp.port
    if port is None:
        return None
    bind_ip = args.work_pose_bind_ip or config.work_pose_udp.bind_ip
    stale_ms = (
        args.work_pose_stale_ms
        if args.work_pose_stale_ms is not None
        else config.work_pose_udp.stale_ms
    )
    client = WorkPoseUdpClient(
        bind_ip=bind_ip,
        port=int(port),
        stale_sec=max(0.0, float(stale_ms)) / 1000.0,
    )
    client.start()
    return client


def _resolve_machine_pose(args: argparse.Namespace, config: CncCompensationConfig) -> MachinePose | None:
    return _resolve_tool_pose(args, config, None)


def _build_tracker_config(args: argparse.Namespace) -> TrackerConfig:
    defaults = TrackerConfig()
    human = bool(args.human_profile)
    return TrackerConfig(
        pose_solver=args.pose_solver,
        smoothing_enabled=not args.no_smoothing,
        translation_alpha=(
            args.translation_alpha
            if args.translation_alpha is not None
            else (0.40 if human else defaults.translation_alpha)
        ),
        rotation_alpha=(
            args.rotation_alpha
            if args.rotation_alpha is not None
            else (0.10 if human else defaults.rotation_alpha)
        ),
        translation_deadband_mm=(
            args.translation_deadband_mm
            if args.translation_deadband_mm is not None
            else (0.0 if human else defaults.translation_deadband_mm)
        ),
        translation_norm_deadband_mm=(
            args.translation_norm_deadband_mm
            if args.translation_norm_deadband_mm is not None
            else (0.0 if human else defaults.translation_norm_deadband_mm)
        ),
        rotation_deadband_deg=(
            args.rotation_deadband_deg
            if args.rotation_deadband_deg is not None
            else (1.8 if human else defaults.rotation_deadband_deg)
        ),
        rotation_max_jump_deg=(
            args.rotation_max_jump_deg
            if args.rotation_max_jump_deg is not None
            else defaults.rotation_max_jump_deg
        ),
        translation_max_jump_mm=(
            args.translation_max_jump_mm
            if args.translation_max_jump_mm is not None
            else (45.0 if human else defaults.translation_max_jump_mm)
        ),
        depth_sample_radius_px=(
            args.depth_sample_radius_px
            if args.depth_sample_radius_px is not None
            else (4 if human else defaults.depth_sample_radius_px)
        ),
        max_landmark_depth_deviation_mm=(
            args.max_landmark_depth_deviation_mm
            if args.max_landmark_depth_deviation_mm is not None
            else (35.0 if human else defaults.max_landmark_depth_deviation_mm)
        ),
    )


def _default_calibration_path() -> str:
    """Prefer cwd config/, else repo-root config/ next to the package source tree."""
    name = "cnc_compensation_quiet.yaml"
    candidates = (
        Path("config") / name,
        Path(__file__).resolve().parents[2] / "config" / name,
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


def _build_compensation_config(args: argparse.Namespace) -> CncCompensationConfig:
    config = load_compensation_config(args.calibration)
    offset_deadband = config.offset_deadband
    if args.human_profile:
        offset_deadband = OffsetDeadbandConfig(
            enabled=True,
            enter_translation_mm=0.4,
            exit_translation_mm=1.0,
            enter_rotation_deg=offset_deadband.enter_rotation_deg,
            exit_rotation_deg=offset_deadband.exit_rotation_deg,
        )
    return CncCompensationConfig(
        machine=config.machine,
        camera_to_machine_rotation=config.camera_to_machine_rotation,
        axis_limits=config.axis_limits,
        safety=config.safety,
        mismatch=config.mismatch,
        offset_deadband=offset_deadband,
        mm_to_axis_unit=config.mm_to_axis_unit,
        deg_to_axis_unit=config.deg_to_axis_unit,
        solver=args.solver or config.solver,
        offset_mode=args.offset_mode or config.offset_mode,
        bc_mode=config.bc_mode,
        bc_camera_rvec_mapping=config.bc_camera_rvec_mapping,
        bc_euler_mapping=config.bc_euler_mapping,
        bc_axis_sign=config.bc_axis_sign,
        machine_pose=config.machine_pose,
        work_pose_udp=config.work_pose_udp,
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
    parser.add_argument(
        "--calibration",
        type=str,
        default=_default_calibration_path(),
        help="CNC compensation YAML (default: config/cnc_compensation_quiet.yaml)",
    )
    parser.add_argument("--machine-pose", type=str, default=None, help="Fallback X,Y,Z,B,C work pose")
    parser.add_argument(
        "--work-pose-udp-port",
        type=int,
        default=None,
        help="Listen for Mach4 active work-coordinate UDP (JSON) on this port",
    )
    parser.add_argument("--work-pose-bind-ip", type=str, default=None)
    parser.add_argument("--work-pose-stale-ms", type=float, default=None)
    parser.add_argument(
        "--require-work-pose",
        action="store_true",
        help="Do not encode offsets until a fresh Mach4 work pose is received",
    )
    parser.add_argument("--capture-baseline-sec", type=float, default=2.0)
    parser.add_argument("--update-period-ms", type=float, default=None)
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Stop after this many seconds (e.g. 7200 for a 2 h stability run)",
    )
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        metavar="PATH_OR_DIR",
        help=(
            "Record head XYZ + yaw/pitch/roll and CNC XYZBC offsets to a timestamped CSV "
            f"(default: {DEFAULT_LOG_DIR.as_posix()}/{DEFAULT_LOG_STEM}_YYYYMMDD_HHMMSS.csv); "
            "give a directory or a .csv path to place it elsewhere"
        ),
    )
    log_group.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Record head pose + XYZBC offsets per tick to this exact CSV path",
    )
    parser.add_argument(
        "--log-rate-hz",
        type=float,
        default=10.0,
        help="Stability log sample rate; decimates the control loop (default: 10)",
    )
    parser.add_argument(
        "--log-timestamped",
        action="store_true",
        help="Append YYYYMMDD_HHMMSS to the --log-csv stem",
    )
    parser.add_argument("--ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--no-ack-watchdog",
        action="store_true",
        help="Do not fault when the controller sends no UDP ACK (view/dry-run only)",
    )
    parser.add_argument("--solver", choices=("kinematic", "decoupled"), default=None)
    parser.add_argument("--offset-mode", choices=("follow", "counter"), default=None)
    parser.add_argument("--pose-solver", choices=("depth-rigid", "pnp", "hybrid"), default="depth-rigid")
    parser.add_argument(
        "--human-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use steadier pose filtering for live human faces (default: on)",
    )
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--translation-alpha", type=float, default=None, help="Translation smoothing factor 0-1; lower is steadier")
    parser.add_argument("--rotation-alpha", type=float, default=None, help="Rotation SLERP factor 0-1; lower is steadier")
    parser.add_argument("--translation-deadband-mm", type=float, default=None, help="Ignore per-axis translation jitter below this (mm)")
    parser.add_argument(
        "--translation-norm-deadband-mm",
        type=float,
        default=None,
        help="Ignore translation jitter when 3D delta norm is below this (mm)",
    )
    parser.add_argument("--rotation-deadband-deg", type=float, default=None, help="Ignore rotation changes below this (deg)")
    parser.add_argument("--rotation-max-jump-deg", type=float, default=None, help="Reject raw rotation jumps above this (deg)")
    parser.add_argument("--translation-max-jump-mm", type=float, default=None, help="Reject raw translation jumps above this (mm)")
    parser.add_argument("--depth-sample-radius-px", type=int, default=None, help="Median depth sampling radius around each landmark")
    parser.add_argument(
        "--max-landmark-depth-deviation-mm",
        type=float,
        default=None,
        help="Drop depth landmarks farther than this from the face depth median",
    )
    parser.add_argument(
        "--step-warn-mm",
        type=float,
        default=1.0,
        help="Warn when one packet moves the offset more than this (0 disables)",
    )
    parser.add_argument(
        "--step-warn-deg",
        type=float,
        default=0.5,
        help="Warn when one packet moves B/C more than this (0 disables)",
    )
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
