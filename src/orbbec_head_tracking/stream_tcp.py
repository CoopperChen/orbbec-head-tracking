from __future__ import annotations

import argparse
import contextlib
import os
import time
from typing import Any

import cv2

from .config import TrackerConfig
from .ethernet_tcp_streamer import TcpJsonlStreamer, TcpJsonlStreamerConfig
from .tracker import OrbbecHeadTracker

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    args = _parse_args()
    config = TrackerConfig(pose_solver=args.pose_solver)
    streamer = TcpJsonlStreamer(
        TcpJsonlStreamerConfig(
            host=args.tcp_host,
            port=args.tcp_port,
            reconnect_interval_sec=args.reconnect_interval_sec,
            connect_timeout_sec=args.connect_timeout_sec,
        )
    )
    with _suppress_native_stderr(not args.verbose):
        with OrbbecHeadTracker(config) as tracker:
            try:
                while True:
                    frame = tracker.read_frame()
                    if frame is None or frame.pose is None:
                        continue
                    pose = frame.pose
                    pitch, yaw, roll = pose.pitch_yaw_roll
                    x_mm, y_mm, z_mm = pose.translation_vector_mm
                    record: dict[str, Any] = {
                        "ts": time.time(),
                        "frame_index": getattr(frame, "frame_index", None),
                        "x_mm": float(x_mm),
                        "y_mm": float(y_mm),
                        "z_mm": float(z_mm),
                        "pitch_deg": float(pitch),
                        "yaw_deg": float(yaw),
                        "roll_deg": float(roll),
                        "confidence": float(getattr(pose, "confidence", 1.0)),
                        "solver": getattr(pose, "solver", args.pose_solver),
                        "valid_depth_count": int(getattr(pose, "valid_depth_count", 0)),
                        "reprojection_error_px": getattr(pose, "reprojection_error_px", None),
                        "smoothed": bool(getattr(pose, "smoothed", False)),
                    }
                    streamer.send_json_obj(record)
            except KeyboardInterrupt:
                return
            finally:
                streamer.close()
                cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Orbbec head pose over TCP as JSONL.")
    parser.add_argument("--tcp-host", type=str, default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5005)
    parser.add_argument("--reconnect-interval-sec", type=float, default=1.0)
    parser.add_argument("--connect-timeout-sec", type=float, default=3.0)
    parser.add_argument("--pose-solver", choices=("depth-rigid", "pnp", "hybrid"), default="depth-rigid")
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
