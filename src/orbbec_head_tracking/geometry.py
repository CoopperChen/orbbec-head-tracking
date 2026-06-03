from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import TrackerConfig
from .constants import EULER_EPSILON, FACE_3D_MODEL, LANDMARK_INDICES, LANDMARK_WEIGHTS


def rotation_matrix_to_euler_degrees(rmat: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0]))
    singular = sy < EULER_EPSILON
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def sample_depth_windows(depth_mm: np.ndarray, points_2d: np.ndarray, radius_px: int) -> np.ndarray:
    height, width = depth_mm.shape
    radius = max(0, int(radius_px))
    out = np.full((len(points_2d),), np.nan, dtype=np.float32)
    for i, (x_float, y_float) in enumerate(points_2d):
        x = int(np.clip(round(float(x_float)), 0, width - 1))
        y = int(np.clip(round(float(y_float)), 0, height - 1))
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size > 0:
            out[i] = float(np.median(valid))
    return out


def pose_confidence(valid_depth_count: int, reprojection_error_px: float | None, config: TrackerConfig) -> float:
    depth_ratio = float(valid_depth_count) / float(len(LANDMARK_INDICES))
    if reprojection_error_px is None:
        return float(np.clip(depth_ratio, 0.0, 1.0))
    repro_score = 1.0 - min(reprojection_error_px / config.max_reprojection_error_px, 1.0)
    return float(np.clip(0.55 * depth_ratio + 0.45 * repro_score, 0.0, 1.0))


def solve_pose_pnp(
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    config: TrackerConfig,
    previous_rotation_vector: np.ndarray | None,
    previous_translation_vector_mm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float] | None:
    use_guess = (
        config.use_previous_pose_guess
        and previous_rotation_vector is not None
        and previous_translation_vector_mm is not None
    )
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        FACE_3D_MODEL,
        points_2d,
        camera_matrix,
        distortion_coefficients,
        rvec=previous_rotation_vector.copy() if use_guess else None,
        tvec=previous_translation_vector_mm.reshape(3, 1).copy() if use_guess else None,
        useExtrinsicGuess=use_guess,
        iterationsCount=config.pnp_iterations_count,
        reprojectionError=config.pnp_reprojection_error,
        confidence=config.pnp_confidence,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(FACE_3D_MODEL, rvec, tvec.reshape(3, 1), camera_matrix, distortion_coefficients)
    err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - points_2d, axis=1)))
    return rvec, tvec, inliers, err


def solve_pose_depth_rigid(
    points_2d: np.ndarray,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    config: TrackerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float] | None:
    sampled = sample_depth_windows(depth_mm, points_2d, config.depth_sample_radius_px)
    valid = np.isfinite(sampled) & (sampled > 0.0)
    if np.count_nonzero(valid) < config.min_depth_points:
        return None
    fx = float(camera_matrix[0, 0]); fy = float(camera_matrix[1, 1]); cx = float(camera_matrix[0, 2]); cy = float(camera_matrix[1, 2])
    obj = []
    cam = []
    idx = []
    for i, (x, y) in enumerate(points_2d):
        if not bool(valid[i]):
            continue
        d = float(sampled[i])
        obj.append(FACE_3D_MODEL[i]); cam.append([(float(x)-cx)*d/fx, (float(y)-cy)*d/fy, d]); idx.append(i)
    object_points = np.asarray(obj, dtype=np.float32)
    camera_points = np.asarray(cam, dtype=np.float32)
    weights = LANDMARK_WEIGHTS[np.asarray(idx, dtype=np.int32)]
    weights = weights / np.sum(weights)
    oc = np.average(object_points, axis=0, weights=weights)
    cc = np.average(camera_points, axis=0, weights=weights)
    cov = ((object_points - oc) * weights[:, None]).T @ ((camera_points - cc) * weights[:, None])
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    tvec = cc.reshape(3, 1) - rot @ oc.reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rot.astype(np.float32))
    projected, _ = cv2.projectPoints(FACE_3D_MODEL, rvec, tvec, camera_matrix, distortion_coefficients)
    err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - points_2d, axis=1)))
    return rvec, tvec, np.asarray(idx, dtype=np.int32).reshape(-1, 1), sampled, len(idx), err


def solve_pose_hybrid(
    points_2d: np.ndarray,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    config: TrackerConfig,
    previous_rotation_vector: np.ndarray | None,
    previous_translation_vector_mm: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float] | None:
    depth = solve_pose_depth_rigid(points_2d, depth_mm, camera_matrix, distortion_coefficients, config)
    if depth is None:
        return None
    rvec0, tvec0, _, sampled, valid_count, _ = depth
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        FACE_3D_MODEL, points_2d, camera_matrix, distortion_coefficients,
        rvec=rvec0, tvec=tvec0, useExtrinsicGuess=True,
        iterationsCount=config.pnp_iterations_count, reprojectionError=config.pnp_reprojection_error,
        confidence=config.pnp_confidence, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(FACE_3D_MODEL, rvec, tvec, camera_matrix, distortion_coefficients)
    err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - points_2d, axis=1)))
    return rvec, tvec, inliers if inliers is not None else np.arange(len(points_2d), dtype=np.int32).reshape(-1, 1), sampled, valid_count, err
