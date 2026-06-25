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


def rotation_matrix_to_quaternion(rmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rmat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([w, x, y, z])
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def align_rotation_matrix(r_new: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
    q_ref = rotation_matrix_to_quaternion(r_ref)
    q_new = rotation_matrix_to_quaternion(r_new)
    if float(np.dot(q_new, q_ref)) < 0.0:
        q_new = -q_new
    return quaternion_to_rotation_matrix(q_new)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_delta = np.asarray(r_a, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(r_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def stabilize_rotation_matrix(
    r_new: np.ndarray,
    r_prev: np.ndarray | None,
    *,
    max_jump_deg: float = 25.0,
) -> np.ndarray:
    r_new = np.asarray(r_new, dtype=np.float64)
    if r_prev is None:
        return r_new
    aligned = align_rotation_matrix(r_new, r_prev)
    if rotation_angle_deg(r_prev, aligned) > float(max_jump_deg):
        return np.asarray(r_prev, dtype=np.float64).copy()
    return aligned


def slerp_rotation_matrices(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    blend = float(np.clip(t, 0.0, 1.0))
    q0 = rotation_matrix_to_quaternion(r0)
    q1 = rotation_matrix_to_quaternion(r1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + blend * (q1 - q0)
        return quaternion_to_rotation_matrix(q)
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-8:
        return np.asarray(r0, dtype=np.float64).copy()
    q = (np.sin((1.0 - blend) * theta) * q0 + np.sin(blend * theta) * q1) / sin_theta
    return quaternion_to_rotation_matrix(q)


def fit_weighted_rigid_rotation(
    object_points: np.ndarray,
    camera_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float64)
    camera_points = np.asarray(camera_points, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(object_points), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    object_centroid = np.average(object_points, axis=0, weights=weights)
    camera_centroid = np.average(camera_points, axis=0, weights=weights)
    object_centered = object_points - object_centroid
    camera_centered = camera_points - camera_centroid
    covariance = (object_centered * weights[:, None]).T @ camera_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    return rotation


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
    rot = fit_weighted_rigid_rotation(object_points, camera_points, weights)
    oc = np.average(object_points, axis=0, weights=weights / np.sum(weights))
    cc = np.average(camera_points, axis=0, weights=weights / np.sum(weights))
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
