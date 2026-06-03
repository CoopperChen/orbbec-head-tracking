from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from pyorbbecsdk import (
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

LANDMARK_MODEL_POINTS = {
    1: [0.0, 0.0, 0.0],
    152: [0.0, -330.0, -65.0],
    33: [-225.0, 170.0, -135.0],
    263: [225.0, 170.0, -135.0],
    61: [-150.0, -150.0, -125.0],
    291: [150.0, -150.0, -125.0],
}
LANDMARK_INDICES = list(LANDMARK_MODEL_POINTS)
FACE_3D_MODEL = np.array(
    [
        LANDMARK_MODEL_POINTS[index]
        for index in LANDMARK_INDICES
    ],
    dtype=np.float32,
)

EULER_EPSILON = 1e-6
AXIS_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],
        [120.0, 0.0, 0.0],
        [0.0, 120.0, 0.0],
        [0.0, 0.0, 120.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class TrackerConfig:
    frame_timeout_ms: int = 100
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.55
    min_tracking_confidence: float = 0.55
    pnp_reprojection_error: float = 8.0
    pnp_confidence: float = 0.99
    pnp_iterations_count: int = 100
    refine_pnp: bool = True
    use_previous_pose_guess: bool = True
    pose_solver: str = "depth-rigid"
    min_depth_points: int = 4
    depth_sample_radius_px: int = 2
    verbose_native_logs: bool = False
    suppress_mediapipe_native_stderr: bool = True
    smoothing_enabled: bool = True
    translation_alpha: float = 0.22
    rotation_alpha: float = 0.28
    translation_deadband_mm: float = 2.5
    rotation_deadband_deg: float = 0.35
    reset_after_missed_frames: int = 12


@dataclass(frozen=True)
class HeadPose:
    rotation_vector: np.ndarray
    translation_vector_mm: np.ndarray
    euler_degrees: tuple[float, float, float]
    landmarks_2d: np.ndarray
    sampled_depth_mm: np.ndarray
    inliers: np.ndarray | None
    smoothed: bool = False

    @property
    def pitch_yaw_roll(self) -> tuple[float, float, float]:
        return self.euler_degrees


@dataclass(frozen=True)
class TrackingFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    pose: HeadPose | None


class PoseSmoother:
    def __init__(
        self,
        translation_alpha: float,
        rotation_alpha: float,
        translation_deadband_mm: float,
        rotation_deadband_deg: float,
    ) -> None:
        self.translation_alpha = float(np.clip(translation_alpha, 0.0, 1.0))
        self.rotation_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
        self.translation_deadband_mm = max(0.0, float(translation_deadband_mm))
        self.rotation_deadband_rad = np.radians(max(0.0, float(rotation_deadband_deg)))
        self.translation_vector_mm: np.ndarray | None = None
        self.rotation_vector: np.ndarray | None = None

    def reset(self) -> None:
        self.translation_vector_mm = None
        self.rotation_vector = None

    def smooth(self, pose: HeadPose) -> HeadPose:
        translation = pose.translation_vector_mm.astype(np.float32).reshape(3)
        rotation = pose.rotation_vector.astype(np.float32).reshape(3, 1)

        if self.translation_vector_mm is None or self.rotation_vector is None:
            self.translation_vector_mm = translation.copy()
            self.rotation_vector = rotation.copy()
            return pose

        smoothed_translation = self._smooth_vector(
            self.translation_vector_mm,
            translation,
            self.translation_alpha,
            self.translation_deadband_mm,
        )
        smoothed_rotation = self._smooth_vector(
            self.rotation_vector.reshape(3),
            rotation.reshape(3),
            self.rotation_alpha,
            self.rotation_deadband_rad,
        ).reshape(3, 1)

        self.translation_vector_mm = smoothed_translation.astype(np.float32)
        self.rotation_vector = smoothed_rotation.astype(np.float32)
        rmat, _ = cv2.Rodrigues(self.rotation_vector)
        return HeadPose(
            rotation_vector=self.rotation_vector.copy(),
            translation_vector_mm=self.translation_vector_mm.copy(),
            euler_degrees=_rotation_matrix_to_euler_degrees(rmat),
            landmarks_2d=pose.landmarks_2d,
            sampled_depth_mm=pose.sampled_depth_mm,
            inliers=pose.inliers,
            smoothed=True,
        )

    @staticmethod
    def _smooth_vector(
        previous: np.ndarray,
        current: np.ndarray,
        alpha: float,
        deadband: float,
    ) -> np.ndarray:
        delta = current - previous
        if deadband > 0.0:
            delta = np.where(np.abs(delta) < deadband, 0.0, delta)
        return previous + alpha * delta


def _camera_matrix_from_profile(color_profile: Any) -> tuple[np.ndarray, np.ndarray]:
    intr = color_profile.get_intrinsic()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    distortion_coefficients = _distortion_from_profile(color_profile)
    return camera_matrix, distortion_coefficients


def _distortion_from_profile(color_profile: Any) -> np.ndarray:
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
        float(getattr(distortion, "k4", 0.0)),
        float(getattr(distortion, "k5", 0.0)),
        float(getattr(distortion, "k6", 0.0)),
    ]
    if any(abs(value) > 0.0 for value in coefficients[5:]):
        return np.asarray(coefficients, dtype=np.float32).reshape(-1, 1)
    return np.asarray(coefficients[:5], dtype=np.float32).reshape(-1, 1)


def _decode_color_frame(color_frame: Any) -> np.ndarray:
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    payload = np.frombuffer(color_frame.get_data(), dtype=np.uint8).copy()

    if _format_matches(fmt, "RGB"):
        rgb = payload.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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


def _format_matches(fmt: Any, name: str) -> bool:
    expected = getattr(OBFormat, name, None)
    return expected is not None and fmt == expected


def _decode_depth_frame_mm(depth_frame: Any) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    if not _format_matches(depth_frame.get_format(), "Y16"):
        raise RuntimeError(f"Unsupported depth frame format: {depth_frame.get_format()}")

    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).copy()
    depth_u16 = depth_raw.reshape((height, width))
    return depth_u16.astype(np.float32) * float(depth_frame.get_depth_scale())


def _landmarks_to_points(
    landmarks: Any,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = []
    for index in LANDMARK_INDICES:
        landmark = landmarks.landmark[index]
        x = np.clip(landmark.x * image_width, 0.0, image_width - 1.0)
        y = np.clip(landmark.y * image_height, 0.0, image_height - 1.0)
        points.append([x, y])
    return np.asarray(points, dtype=np.float32)


def _sample_depth(depth_mm: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    height, width = depth_mm.shape
    samples = []
    for x_float, y_float in points_2d:
        x = int(np.clip(round(float(x_float)), 0, width - 1))
        y = int(np.clip(round(float(y_float)), 0, height - 1))
        samples.append(depth_mm[y, x])
    return np.asarray(samples, dtype=np.float32)


def _sample_depth_window(
    depth_mm: np.ndarray,
    x_float: float,
    y_float: float,
    radius_px: int,
) -> float:
    height, width = depth_mm.shape
    x = int(np.clip(round(float(x_float)), 0, width - 1))
    y = int(np.clip(round(float(y_float)), 0, height - 1))
    radius = max(0, int(radius_px))
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _points_to_camera_3d(
    points_2d: np.ndarray,
    depth_mm: np.ndarray,
    camera_matrix: np.ndarray,
    radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    object_points = []
    camera_points = []
    sampled_depth = []

    for index, (x_float, y_float) in enumerate(points_2d):
        depth_value = _sample_depth_window(depth_mm, float(x_float), float(y_float), radius_px)
        sampled_depth.append(depth_value)
        if not np.isfinite(depth_value) or depth_value <= 0.0:
            continue
        x_camera = (float(x_float) - cx) * depth_value / fx
        y_camera = (float(y_float) - cy) * depth_value / fy
        object_points.append(FACE_3D_MODEL[index])
        camera_points.append([x_camera, y_camera, depth_value])

    return (
        np.asarray(object_points, dtype=np.float32),
        np.asarray(camera_points, dtype=np.float32),
        np.asarray(sampled_depth, dtype=np.float32),
    )


def _fit_rigid_transform(
    object_points: np.ndarray,
    camera_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_centroid = object_points.mean(axis=0)
    camera_centroid = camera_points.mean(axis=0)
    object_centered = object_points - object_centroid
    camera_centered = camera_points - camera_centroid
    covariance = object_centered.T @ camera_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    translation = camera_centroid.reshape(3, 1) - rotation @ object_centroid.reshape(3, 1)
    rotation_vector, _ = cv2.Rodrigues(rotation.astype(np.float32))
    return rotation_vector.astype(np.float32), translation.astype(np.float32)


def _rotation_matrix_to_euler_degrees(rmat: np.ndarray) -> tuple[float, float, float]:
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

    return (
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
        float(np.degrees(roll)),
    )


class OrbbecHeadTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.pipeline: Pipeline | None = None
        self.align_filter: AlignFilter | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion_coefficients: np.ndarray | None = None
        self.face_mesh: Any | None = None
        self.pose_smoother = PoseSmoother(
            self.config.translation_alpha,
            self.config.rotation_alpha,
            self.config.translation_deadband_mm,
            self.config.rotation_deadband_deg,
        )
        self.missed_pose_count = 0
        self.previous_raw_rotation_vector: np.ndarray | None = None
        self.previous_raw_translation_vector_mm: np.ndarray | None = None

    def start(self) -> None:
        if self.pipeline is not None:
            return

        pipeline = Pipeline()
        sdk_config = Config()
        _configure_orbbec_logging(self.config.verbose_native_logs)
        color_profile = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_default_video_stream_profile()
        depth_profile = pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()

        sdk_config.enable_stream(color_profile)
        sdk_config.enable_stream(depth_profile)
        sdk_config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )

        pipeline.enable_frame_sync()
        pipeline.start(sdk_config)

        self.pipeline = pipeline
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.camera_matrix, self.distortion_coefficients = _camera_matrix_from_profile(
            color_profile
        )
        with _suppress_native_stderr(
            self.config.suppress_mediapipe_native_stderr
            and not self.config.verbose_native_logs
        ):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )

    def stop(self) -> None:
        self.pose_smoother.reset()
        self.missed_pose_count = 0
        self.previous_raw_rotation_vector = None
        self.previous_raw_translation_vector_mm = None
        face_mesh = self.face_mesh
        self.face_mesh = None
        if face_mesh is not None:
            face_mesh.close()

        pipeline = self.pipeline
        self.pipeline = None
        self.align_filter = None
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> OrbbecHeadTracker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def read_pose(self) -> HeadPose | None:
        frame = self.read_frame()
        if frame is None:
            return None
        return frame.pose

    def read_frame(self) -> TrackingFrame | None:
        if (
            self.pipeline is None
            or self.align_filter is None
            or self.camera_matrix is None
            or self.distortion_coefficients is None
            or self.face_mesh is None
        ):
            raise RuntimeError("Tracker must be started before reading poses")

        frames = self.pipeline.wait_for_frames(self.config.frame_timeout_ms)
        if frames is None:
            return None

        aligned = self.align_filter.process(frames)
        if not aligned:
            return None

        frame_set = aligned.as_frame_set()
        color_frame = frame_set.get_color_frame()
        depth_frame = frame_set.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        color_bgr = _decode_color_frame(color_frame)
        depth_mm = _decode_depth_frame_mm(depth_frame)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        with _suppress_native_stderr(
            self.config.suppress_mediapipe_native_stderr
            and not self.config.verbose_native_logs
        ):
            result = self.face_mesh.process(color_rgb)
        if not result.multi_face_landmarks:
            self._mark_pose_missed()
            return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=None)

        image_height, image_width = color_bgr.shape[:2]
        points_2d = _landmarks_to_points(
            result.multi_face_landmarks[0],
            image_width,
            image_height,
        )
        if self.config.pose_solver == "depth-rigid":
            object_points, camera_points, sampled_depth = _points_to_camera_3d(
                points_2d,
                depth_mm,
                self.camera_matrix,
                self.config.depth_sample_radius_px,
            )
            if len(camera_points) < self.config.min_depth_points:
                self._mark_pose_missed()
                return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=None)
            rvec, tvec = _fit_rigid_transform(object_points, camera_points)
            inliers = np.arange(len(camera_points), dtype=np.int32).reshape(-1, 1)
        else:
            sampled_depth = _sample_depth(depth_mm, points_2d)
            rvec_guess = self.previous_raw_rotation_vector
            tvec_guess = self.previous_raw_translation_vector_mm
            use_guess = (
                self.config.use_previous_pose_guess
                and rvec_guess is not None
                and tvec_guess is not None
            )
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                FACE_3D_MODEL,
                points_2d,
                self.camera_matrix,
                self.distortion_coefficients,
                rvec=rvec_guess.copy() if use_guess else None,
                tvec=tvec_guess.reshape(3, 1).copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                iterationsCount=self.config.pnp_iterations_count,
                reprojectionError=self.config.pnp_reprojection_error,
                confidence=self.config.pnp_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                self._mark_pose_missed()
                return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=None)

            if self.config.refine_pnp and inliers is not None and len(inliers) >= 6:
                inlier_indices = inliers.reshape(-1)
                object_inliers = FACE_3D_MODEL[inlier_indices]
                image_inliers = points_2d[inlier_indices]
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_inliers,
                    image_inliers,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    rvec,
                    tvec,
                )

        rmat, _ = cv2.Rodrigues(rvec)
        euler_degrees = _rotation_matrix_to_euler_degrees(rmat)
        self.previous_raw_rotation_vector = rvec.astype(np.float32)
        self.previous_raw_translation_vector_mm = tvec.reshape(3).astype(np.float32)
        pose = HeadPose(
            rotation_vector=self.previous_raw_rotation_vector,
            translation_vector_mm=self.previous_raw_translation_vector_mm,
            euler_degrees=euler_degrees,
            landmarks_2d=points_2d,
            sampled_depth_mm=sampled_depth,
            inliers=inliers,
        )
        self.missed_pose_count = 0
        if self.config.smoothing_enabled:
            pose = self.pose_smoother.smooth(pose)
        return TrackingFrame(color_bgr=color_bgr, depth_mm=depth_mm, pose=pose)

    def _mark_pose_missed(self) -> None:
        self.missed_pose_count += 1
        if self.missed_pose_count >= self.config.reset_after_missed_frames:
            self.pose_smoother.reset()
            self.previous_raw_rotation_vector = None
            self.previous_raw_translation_vector_mm = None


def draw_pose_overlay(
    frame: TrackingFrame,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> np.ndarray:
    canvas = frame.color_bgr.copy()
    pose = frame.pose
    if pose is None:
        _draw_status_panel(canvas, "No face pose", None)
        return canvas

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
    origin = _as_image_point(axis_points[0])
    x_axis = _as_image_point(axis_points[1])
    y_axis = _as_image_point(axis_points[2])
    z_axis = _as_image_point(axis_points[3])
    cv2.arrowedLine(canvas, origin, x_axis, (0, 0, 255), 3, tipLength=0.18)
    cv2.arrowedLine(canvas, origin, y_axis, (0, 255, 0), 3, tipLength=0.18)
    cv2.arrowedLine(canvas, origin, z_axis, (255, 0, 0), 3, tipLength=0.18)
    cv2.putText(canvas, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    _draw_status_panel(canvas, "Tracking", pose)
    return canvas


def _as_image_point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _draw_status_panel(
    canvas: np.ndarray,
    status: str,
    pose: HeadPose | None,
) -> None:
    panel_width = 360
    panel_height = 136 if pose is not None else 56
    overlay = canvas.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 12 + panel_height), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(canvas, (12, 12), (12 + panel_width, 12 + panel_height), (80, 180, 255), 1)
    cv2.putText(canvas, status, (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 245, 255), 2)

    if pose is None:
        return

    pitch, yaw, roll = pose.pitch_yaw_roll
    x_mm, y_mm, z_mm = pose.translation_vector_mm
    support_count = 0 if pose.inliers is None else int(len(pose.inliers))
    rows = [
        f"X {x_mm:7.1f} mm   Y {y_mm:7.1f} mm   Z {z_mm:7.1f} mm",
        f"Pitch {pitch:6.2f}   Yaw {yaw:6.2f}   Roll {roll:6.2f}",
        f"Depth samples {np.nanmean(pose.sampled_depth_mm):7.1f} mm avg"
        f"   support {support_count:02d}/{len(LANDMARK_INDICES):02d}"
        f"   {'smoothed' if pose.smoothed else 'raw'}",
    ]
    for offset, text in enumerate(rows):
        cv2.putText(
            canvas,
            text,
            (28, 78 + offset * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 236, 244),
            1,
            lineType=cv2.LINE_AA,
        )


def main() -> None:
    args = _parse_args()
    config = TrackerConfig(
        frame_timeout_ms=args.frame_timeout_ms,
        verbose_native_logs=args.verbose,
        smoothing_enabled=not args.no_smoothing,
        translation_alpha=args.translation_alpha,
        rotation_alpha=args.rotation_alpha,
        translation_deadband_mm=args.translation_deadband_mm,
        rotation_deadband_deg=args.rotation_deadband_deg,
        reset_after_missed_frames=args.reset_after_missed_frames,
        refine_pnp=not args.no_pnp_refine,
        use_previous_pose_guess=not args.no_previous_pose_guess,
        pose_solver=args.pose_solver,
        min_depth_points=args.min_depth_points,
        depth_sample_radius_px=args.depth_sample_radius_px,
    )
    with _suppress_native_stderr(not args.verbose):
        with OrbbecHeadTracker(config) as tracker:
            try:
                if args.view:
                    _run_viewer(tracker, args.window_name)
                else:
                    _run_text_loop(tracker)
            except KeyboardInterrupt:
                return
            finally:
                cv2.destroyAllWindows()


def _run_text_loop(tracker: OrbbecHeadTracker) -> None:
    while True:
        pose = tracker.read_pose()
        if pose is None:
            continue
        pitch, yaw, roll = pose.pitch_yaw_roll
        x_mm, y_mm, z_mm = pose.translation_vector_mm
        print(
            "pose "
            f"x={x_mm:8.1f}mm y={y_mm:8.1f}mm z={z_mm:8.1f}mm "
            f"pitch={pitch:7.2f} yaw={yaw:7.2f} roll={roll:7.2f}",
            flush=True,
        )


def _run_viewer(tracker: OrbbecHeadTracker, window_name: str) -> None:
    if tracker.camera_matrix is None or tracker.distortion_coefficients is None:
        raise RuntimeError("Tracker calibration is unavailable")

    depth_window_name = f"{window_name} Depth"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)
    while True:
        frame = tracker.read_frame()
        if frame is None:
            continue
        visualization = draw_pose_overlay(
            frame,
            tracker.camera_matrix,
            tracker.distortion_coefficients,
        )
        cv2.imshow(window_name, visualization)
        cv2.imshow(depth_window_name, colorize_depth_mm(frame.depth_mm))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return


def colorize_depth_mm(
    depth_mm: np.ndarray,
    min_depth_mm: float = 250.0,
    max_depth_mm: float = 2500.0,
) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
    clipped = np.clip(depth_mm, min_depth_mm, max_depth_mm)
    normalized = ((clipped - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255.0)
    depth_u8 = normalized.astype(np.uint8)
    colorized = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    cv2.putText(
        colorized,
        f"Depth {min_depth_mm:.0f}-{max_depth_mm:.0f} mm",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return colorized


def _configure_orbbec_logging(verbose: bool) -> None:
    level = OBLogLevel.WARNING if verbose else OBLogLevel.FATAL
    context = Context()
    context.set_logger_level(level)
    context.set_logger_to_console(level)


@contextlib.contextmanager
def _suppress_native_stderr(enabled: bool) -> Any:
    if not enabled:
        yield
        return

    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    try:
        with Path(os.devnull).open("w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Orbbec Gemini 2L 6-DoF head tracking."
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=TrackerConfig.frame_timeout_ms,
        help="Frame wait timeout in milliseconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show native Orbbec SDK warning/error logs.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Show a live RGB visualization window with pose overlay.",
    )
    parser.add_argument(
        "--window-name",
        default="Orbbec Head Pose",
        help="Title for the visualization window.",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal pose smoothing and show raw PnP estimates.",
    )
    parser.add_argument(
        "--translation-alpha",
        type=float,
        default=TrackerConfig.translation_alpha,
        help="Translation smoothing factor from 0 to 1; lower is steadier.",
    )
    parser.add_argument(
        "--rotation-alpha",
        type=float,
        default=TrackerConfig.rotation_alpha,
        help="Rotation smoothing factor from 0 to 1; lower is steadier.",
    )
    parser.add_argument(
        "--translation-deadband-mm",
        type=float,
        default=TrackerConfig.translation_deadband_mm,
        help="Ignore per-axis translation jitter below this many millimeters.",
    )
    parser.add_argument(
        "--rotation-deadband-deg",
        type=float,
        default=TrackerConfig.rotation_deadband_deg,
        help="Ignore rotation-vector jitter below this many degrees.",
    )
    parser.add_argument(
        "--reset-after-missed-frames",
        type=int,
        default=TrackerConfig.reset_after_missed_frames,
        help="Reset smoothing after this many consecutive missed pose frames.",
    )
    parser.add_argument(
        "--pose-solver",
        choices=("depth-rigid", "pnp"),
        default=TrackerConfig.pose_solver,
        help="Pose method: depth-rigid fits 3D landmarks from depth; pnp uses 2D landmarks only.",
    )
    parser.add_argument(
        "--min-depth-points",
        type=int,
        default=TrackerConfig.min_depth_points,
        help="Minimum valid depth landmarks required by the depth-rigid solver.",
    )
    parser.add_argument(
        "--depth-sample-radius-px",
        type=int,
        default=TrackerConfig.depth_sample_radius_px,
        help="Median depth sampling radius around each landmark.",
    )
    parser.add_argument(
        "--no-pnp-refine",
        action="store_true",
        help="Disable inlier-only solvePnPRefineLM refinement after RANSAC.",
    )
    parser.add_argument(
        "--no-previous-pose-guess",
        action="store_true",
        help="Disable using the previous pose as the next PnP initial guess.",
    )
    return parser.parse_args()


def viewer_main() -> None:
    with _patched_argv_for_viewer():
        main()


@contextlib.contextmanager
def _patched_argv_for_viewer() -> Any:
    import sys

    original = sys.argv[:]
    if "--view" not in sys.argv:
        sys.argv.append("--view")
    try:
        yield
    finally:
        sys.argv[:] = original


if __name__ == "__main__":
    main()

