from __future__ import annotations

import cv2
import numpy as np

from .constants import AXIS_3D_MODEL, LANDMARK_INDICES
from .types import HeadPose, TrackingFrame


def draw_pose_overlay(frame: TrackingFrame, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    canvas = frame.color_bgr.copy()
    if frame.pose is None:
        return canvas
    pose = frame.pose
    for point in pose.landmarks_2d:
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(canvas, center, 4, (0, 220, 255), -1, lineType=cv2.LINE_AA)
    projected_axes, _ = cv2.projectPoints(
        AXIS_3D_MODEL,
        pose.rotation_vector,
        pose.translation_vector_mm.reshape(3, 1),
        camera_matrix,
        distortion_coefficients,
    )
    axis_points = projected_axes.reshape(-1, 2)
    origin = (int(axis_points[0, 0]), int(axis_points[0, 1]))
    x_axis = (int(axis_points[1, 0]), int(axis_points[1, 1]))
    y_axis = (int(axis_points[2, 0]), int(axis_points[2, 1]))
    z_axis = (int(axis_points[3, 0]), int(axis_points[3, 1]))
    cv2.arrowedLine(canvas, origin, x_axis, (0, 0, 255), 3, tipLength=0.18)
    cv2.arrowedLine(canvas, origin, y_axis, (0, 255, 0), 3, tipLength=0.18)
    cv2.arrowedLine(canvas, origin, z_axis, (255, 0, 0), 3, tipLength=0.18)
    return canvas


def colorize_depth_mm(depth_mm: np.ndarray, min_depth_mm: float = 250.0, max_depth_mm: float = 2500.0) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
    clipped = np.clip(depth_mm, min_depth_mm, max_depth_mm)
    normalized = ((clipped - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255.0)
    depth_u8 = normalized.astype(np.uint8)
    colorized = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    return colorized
