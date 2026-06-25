"""Shared Orbbec capture for temporary pipeline stage viewers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running from scripts/pipeline_demos without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from orbbec_head_tracking.frames import camera_matrix_from_profile
from orbbec_head_tracking.orbbec_sdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


def decode_color_frame(color_frame: Any) -> np.ndarray:
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    payload = np.frombuffer(color_frame.get_data(), dtype=np.uint8).copy()
    if _format_matches(fmt, "RGB"):
        return cv2.cvtColor(payload.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    if _format_matches(fmt, "BGR"):
        return payload.reshape((height, width, 3))
    if _format_matches(fmt, "MJPG"):
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode MJPG color frame")
        return image
    if _format_matches(fmt, "YUYV"):
        yuyv = payload.reshape((height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if _format_matches(fmt, "I420"):
        i420 = payload.reshape((height * 3 // 2, width))
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    if _format_matches(fmt, "NV12"):
        nv12 = payload.reshape((height * 3 // 2, width))
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    raise RuntimeError(f"Unsupported color frame format: {fmt}")


def decode_depth_frame_mm(depth_frame: Any) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    if not _format_matches(depth_frame.get_format(), "Y16"):
        raise RuntimeError(f"Unsupported depth frame format: {depth_frame.get_format()}")
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).copy()
    return depth_raw.reshape((height, width)).astype(np.float32) * float(depth_frame.get_depth_scale())


def colorize_depth_mm(
    depth_mm: np.ndarray,
    min_depth_mm: float = 250.0,
    max_depth_mm: float = 2500.0,
) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
    clipped = np.clip(depth_mm, min_depth_mm, max_depth_mm)
    normalized = (clipped - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255.0
    depth_u8 = normalized.astype(np.uint8)
    colorized = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    return colorized


def _format_matches(fmt: Any, name: str) -> bool:
    expected = getattr(OBFormat, name, None)
    return expected is not None and fmt == expected


@dataclass(frozen=True)
class PipelineSnapshot:
    color_bgr: np.ndarray
    depth_native_mm: np.ndarray
    depth_aligned_mm: np.ndarray

    @property
    def depth_native_vis(self) -> np.ndarray:
        return _resize_to_height(colorize_depth_mm(self.depth_native_mm), self.color_bgr.shape[0])

    @property
    def depth_aligned_vis(self) -> np.ndarray:
        return colorize_depth_mm(self.depth_aligned_mm)


class PipelineCapture:
    def __init__(self, frame_timeout_ms: int = 100) -> None:
        self.frame_timeout_ms = frame_timeout_ms
        self.pipeline: Pipeline | None = None
        self.align_filter: AlignFilter | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion_coefficients: np.ndarray | None = None

    def start(self) -> None:
        if self.pipeline is not None:
            return
        _configure_orbbec_logging(verbose=False)
        pipeline = Pipeline()
        config = Config()
        color_profile = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_default_video_stream_profile()
        depth_profile = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        pipeline.start(config)
        self.pipeline = pipeline
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.camera_matrix, self.distortion_coefficients = camera_matrix_from_profile(color_profile)

    def stop(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        self.align_filter = None
        self.camera_matrix = None
        self.distortion_coefficients = None
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> PipelineCapture:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def read(self) -> PipelineSnapshot | None:
        if self.pipeline is None or self.align_filter is None:
            raise RuntimeError("PipelineCapture.start() must be called first")
        frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
        if frames is None:
            return None

        raw_set = frames.as_frame_set()
        raw_depth_frame = raw_set.get_depth_frame()
        if raw_depth_frame is None:
            return None
        depth_native_mm = decode_depth_frame_mm(raw_depth_frame)

        aligned = self.align_filter.process(frames)
        if not aligned:
            return None
        aligned_set = aligned.as_frame_set()
        color_frame = aligned_set.get_color_frame()
        depth_frame = aligned_set.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color_bgr = decode_color_frame(color_frame)
        depth_aligned_mm = decode_depth_frame_mm(depth_frame)
        return PipelineSnapshot(
            color_bgr=color_bgr,
            depth_native_mm=depth_native_mm,
            depth_aligned_mm=depth_aligned_mm,
        )


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / float(height)
    new_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_NEAREST)


def blend_rgb_depth(color_bgr: np.ndarray, depth_mm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    depth_vis = colorize_depth_mm(depth_mm)
    valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
    blended = color_bgr.copy()
    mix = cv2.addWeighted(color_bgr, 1.0 - alpha, depth_vis, alpha, 0.0)
    blended[valid] = mix[valid]
    return blended


def run_view_loop(title: str, render_fn: Any) -> None:
    """Press Q or Esc to quit."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        with PipelineCapture() as capture:
            while True:
                snapshot = capture.read()
                if snapshot is None:
                    continue
                canvas = render_fn(snapshot)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return
    finally:
        cv2.destroyAllWindows()


def _configure_orbbec_logging(verbose: bool) -> None:
    try:
        context = Context()
        context.set_logger_level(OBLogLevel.DEBUG if verbose else OBLogLevel.ERROR)
    except Exception:
        pass
