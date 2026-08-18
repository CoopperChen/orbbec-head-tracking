from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cnc_mismatch import CncMismatchTracker
from .cnc_offset_encoder import CncOffsetEncoder, CncUserOffset
from .cnc_protocol import UserOffsetMessage
from .cnc_safety import CncSafetyGuards, SafetyDecision
from .cnc_udp_streamer import CncUdpStreamer, CncUdpStreamerConfig
from .pipeline_timing import (
    LoopTimingSample,
    VisionStageTiming,
    format_summary_table,
    summarize_timing,
    write_timing_csv,
)
from .stream_cnc import (
    _build_compensation_config,
    _build_tracker_config,
    _default_calibration_path,
    _lock_baseline,
    _open_work_pose_client,
    _resolve_tool_pose,
    _suppress_native_stderr,
)
from .tracker import OrbbecHeadTracker, TrackingFrame

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _timed_cnc_stages(
    *,
    frame: TrackingFrame | None,
    encoder: CncOffsetEncoder,
    mismatch: CncMismatchTracker,
    safety: CncSafetyGuards,
    streamer: CncUdpStreamer,
    args: argparse.Namespace,
    comp_config: Any,
    work_pose_client: Any,
    period_sec: float,
    now: float,
    baseline_deadline: float,
    baseline_translations: list[np.ndarray],
    baseline_rvecs: list[np.ndarray],
) -> tuple[float, float, SafetyDecision, bool, bool]:
    tracking_ok = frame is not None and frame.pose is not None
    confidence = 0.0
    head_speed = 0.0
    proposed = CncUserOffset.zero()
    target = CncUserOffset.zero()
    mismatch_report = None
    require_live_work_pose = args.require_work_pose or comp_config.work_pose_udp.require_live
    work_pose_missing = False
    encode_ms = 0.0
    head_speed = 0.0
    d_t_machine = np.zeros(3, dtype=np.float64)
    encoded = False

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
                encode_start = time.perf_counter()
                proposed = encoder.encode(pose, machine_pose=tool_pose)
                encode_ms = (time.perf_counter() - encode_start) * 1000.0
                head_speed = safety.estimate_head_speed_mm_s(
                    np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3),
                    now,
                )
                d_t_machine = encoder.head_delta_machine_mm(pose)
                encoded = True
            else:
                work_pose_missing = require_live_work_pose
        else:
            target = safety.last_offset
    else:
        target = safety.last_offset

    safety_start = time.perf_counter()
    if encoded:
        target, mismatch_report = mismatch.target(
            proposed,
            safety.last_offset,
            dt_sec=period_sec,
            head_speed_mm_s=head_speed,
            d_t_machine=d_t_machine,
            preserve_sent=safety.in_recovery,
        )
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
            snap=(mismatch_report is not None and mismatch_report.mode == "snap"),
        )
    safety_rate_limit_ms = (time.perf_counter() - safety_start) * 1000.0

    if decision.action == "hold_last":
        mismatch.notify_command_hold(decision.reason)

    return encode_ms, safety_rate_limit_ms, decision, tracking_ok, frame is not None


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
            ack_watchdog_enabled=args.ack_watchdog,
            motor_map=comp_config.motor_map,
        )
    )
    work_pose_client = _open_work_pose_client(args, comp_config)

    period_sec = comp_config.update_period_ms / 1000.0
    baseline_deadline = time.monotonic() + max(0.0, args.capture_baseline_sec)
    baseline_translations: list[np.ndarray] = []
    baseline_rvecs: list[np.ndarray] = []
    samples: list[LoopTimingSample] = []
    prev_loop_start = time.perf_counter()

    print(
        f"Pipeline benchmark: {args.loops} loops @ {comp_config.update_period_ms:.1f} ms "
        f"(warmup {args.warmup}, output {args.output})"
    )

    with _suppress_native_stderr(not args.verbose):
        with OrbbecHeadTracker(tracker_config) as tracker:
            streamer.connect()
            try:
                for loop_index in range(args.loops):
                    loop_start = time.perf_counter()
                    loop_wall_time = time.time()
                    loop_period_ms = (loop_start - prev_loop_start) * 1000.0
                    prev_loop_start = loop_start
                    now = time.monotonic()

                    vision_timing = VisionStageTiming()
                    frame = tracker.read_frame(timing=vision_timing)

                    encode_ms, safety_rate_limit_ms, decision, tracking_ok, face_detected = (
                        _timed_cnc_stages(
                            frame=frame,
                            encoder=encoder,
                            mismatch=mismatch,
                            safety=safety,
                            streamer=streamer,
                            args=args,
                            comp_config=comp_config,
                            work_pose_client=work_pose_client,
                            period_sec=period_sec,
                            now=now,
                            baseline_deadline=baseline_deadline,
                            baseline_translations=baseline_translations,
                            baseline_rvecs=baseline_rvecs,
                        )
                    )

                    message = UserOffsetMessage.from_xyzbc(
                        *decision.offset.as_tuple(),
                        motor_map=comp_config.motor_map,
                    )
                    udp_start = time.perf_counter()
                    streamer.send_message(message)
                    udp_ms = (time.perf_counter() - udp_start) * 1000.0

                    loop_total_ms = (time.perf_counter() - loop_start) * 1000.0
                    samples.append(
                        LoopTimingSample(
                            loop_index=loop_index,
                            wall_time_sec=loop_wall_time,
                            loop_period_ms=loop_period_ms,
                            frame_acquire_align_ms=vision_timing.frame_acquire_align_ms,
                            landmark_ms=vision_timing.landmark_ms,
                            pose_estimate_ms=vision_timing.pose_estimate_ms,
                            temporal_filter_ms=vision_timing.temporal_filter_ms,
                            encode_ms=encode_ms,
                            safety_rate_limit_ms=safety_rate_limit_ms,
                            udp_ms=udp_ms,
                            loop_total_ms=loop_total_ms,
                            tracking_ok=tracking_ok,
                            baseline_ready=encoder.baseline_ready,
                            face_detected=face_detected,
                        )
                    )

                    if args.verbose and loop_index % max(1, args.loops // 10) == 0:
                        print(
                            f"[{loop_index:4d}] total={loop_total_ms:6.2f} ms "
                            f"frame={vision_timing.frame_acquire_align_ms:5.2f} "
                            f"mp={vision_timing.landmark_ms:5.2f} "
                            f"pose={vision_timing.pose_estimate_ms:5.2f} "
                            f"smooth={vision_timing.temporal_filter_ms:5.2f} "
                            f"encode={encode_ms:5.2f} safety={safety_rate_limit_ms:5.2f} "
                            f"udp={udp_ms:5.2f}"
                        )

                    sleep_until = loop_start + period_sec
                    remaining = sleep_until - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)

            except KeyboardInterrupt:
                print("\nInterrupted; writing collected samples.")
            finally:
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
                streamer.close()
                if work_pose_client is not None:
                    work_pose_client.close()

    output_path = Path(args.output)
    write_timing_csv(output_path, samples)
    print(f"Wrote {len(samples)} samples to {output_path}")

    analysis_samples = samples[args.warmup :] if args.warmup > 0 else samples
    if analysis_samples:
        summary = summarize_timing(analysis_samples)
        summary_text = format_summary_table(summary)
        if args.summary:
            summary_path = Path(args.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_text + "\n", encoding="utf-8")
            print(f"Wrote summary to {summary_path}")
        print("\nTiming summary (post-warmup):")
        print(summary_text)
    else:
        print("No samples collected for summary.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark per-stage latency of the Orbbec CNC pipeline and export raw CSV "
            "for statistical analysis."
        )
    )
    parser.add_argument("--loops", type=int, default=300, help="Number of timed control loops")
    parser.add_argument("--warmup", type=int, default=30, help="Exclude first N loops from summary")
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_timing.csv",
        help="Raw per-loop CSV output path",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional CSV summary table path (median/P95 printed to stdout regardless)",
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
    parser.add_argument("--machine-pose", type=str, default=None)
    parser.add_argument("--work-pose-udp-port", type=int, default=None)
    parser.add_argument("--work-pose-bind-ip", type=str, default=None)
    parser.add_argument("--work-pose-stale-ms", type=float, default=None)
    parser.add_argument("--require-work-pose", action="store_true")
    parser.add_argument("--capture-baseline-sec", type=float, default=2.0)
    parser.add_argument(
        "--update-period-ms",
        type=float,
        default=33.0,
        help="Control loop period (default 33 ms ~30 Hz; processing is timed independently)",
    )
    parser.add_argument("--ack-timeout-ms", type=float, default=2000.0)
    parser.add_argument(
        "--ack-watchdog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable UDP ACK watchdog during benchmark (default: off)",
    )
    parser.add_argument("--solver", choices=("kinematic", "decoupled"), default=None)
    parser.add_argument("--offset-mode", choices=("follow", "counter"), default=None)
    parser.add_argument("--pose-solver", choices=("depth-rigid", "pnp", "hybrid"), default="depth-rigid")
    parser.add_argument(
        "--human-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--translation-alpha", type=float, default=None)
    parser.add_argument("--rotation-alpha", type=float, default=None)
    parser.add_argument("--translation-deadband-mm", type=float, default=None)
    parser.add_argument("--translation-norm-deadband-mm", type=float, default=None)
    parser.add_argument("--rotation-deadband-deg", type=float, default=None)
    parser.add_argument("--rotation-max-jump-deg", type=float, default=None)
    parser.add_argument("--translation-max-jump-mm", type=float, default=None)
    parser.add_argument("--depth-sample-radius-px", type=int, default=None)
    parser.add_argument("--max-landmark-depth-deviation-mm", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
