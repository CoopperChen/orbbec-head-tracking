from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .constants import LANDMARK_INDICES

PoseSolver = Literal["depth-rigid", "pnp", "hybrid"]
FrameAggregateMode = Literal["full", "any"]


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
    pose_solver: PoseSolver = "depth-rigid"
    min_depth_points: int = 4
    depth_sample_radius_px: int = 2
    max_reprojection_error_px: float = 12.0
    max_landmark_depth_deviation_mm: float = 120.0
    refine_with_pnp_after_depth: bool = True
    face_mesh_input_scale: float = 1.0
    frame_aggregate_mode: FrameAggregateMode = "full"
    color_width: int | None = None
    color_height: int | None = None
    verbose_native_logs: bool = False
    suppress_mediapipe_native_stderr: bool = True
    smoothing_enabled: bool = True
    translation_alpha: float = 0.22
    rotation_alpha: float = 0.28
    translation_deadband_mm: float = 2.5
    rotation_deadband_deg: float = 0.35
    reset_after_missed_frames: int = 12

    def __post_init__(self) -> None:
        if self.frame_timeout_ms <= 0:
            raise ValueError("frame_timeout_ms must be positive")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be in (0, 1]")
        if not 0.0 < self.min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be in (0, 1]")
        if self.pnp_reprojection_error <= 0.0:
            raise ValueError("pnp_reprojection_error must be positive")
        if not 0.0 < self.pnp_confidence < 1.0:
            raise ValueError("pnp_confidence must be in (0, 1)")
        if self.pnp_iterations_count <= 0:
            raise ValueError("pnp_iterations_count must be positive")
        if self.pose_solver not in ("depth-rigid", "pnp", "hybrid"):
            raise ValueError("pose_solver must be depth-rigid, pnp, or hybrid")
        if not 4 <= self.min_depth_points <= len(LANDMARK_INDICES):
            raise ValueError("min_depth_points must be between 4 and landmark count")
