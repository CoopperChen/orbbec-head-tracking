from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from pyorbbecsdk import OBFormat


def camera_matrix_from_profile(color_profile: Any) -> tuple[np.ndarray, np.ndarray]:
    intr = color_profile.get_intrinsic()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    distortion_coefficients = distortion_from_profile(color_profile)
    return camera_matrix, distortion_coefficients


def distortion_from_profile(color_profile: Any) -> np.ndarray:
    get_distortion = getattr(color_profile, "get_distortion", None)
    if get_distortion is None:
        return np.zeros((5, 1), dtype=np.float32)
    distortion = get_distortion()
    coefficients = [
        float(getattr(distortion, "k1", 0.0)),
        float(getattr(distortion, "k2", 0.0)),
        float(getattr(distortion, "p1", 0.0)),
        float(getattr(distortion, "p2", 0.0)),
        float(getattr(distortion, "k3", 0.0)),
    ]
    return np.asarray(coefficients, dtype=np.float32).reshape(-1, 1)


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
    raise RuntimeError(f"Unsupported color frame format: {fmt}")


def decode_depth_frame_mm(depth_frame: Any) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    if not _format_matches(depth_frame.get_format(), "Y16"):
        raise RuntimeError(f"Unsupported depth frame format: {depth_frame.get_format()}")
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).copy()
    return depth_raw.reshape((height, width)).astype(np.float32) * float(depth_frame.get_depth_scale())


def _format_matches(fmt: Any, name: str) -> bool:
    expected = getattr(OBFormat, name, None)
    return expected is not None and fmt == expected
