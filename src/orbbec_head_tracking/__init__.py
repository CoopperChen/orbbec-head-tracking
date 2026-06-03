"""Orbbec Gemini 2L 6-DoF head tracking package."""

from .constants import FACE_3D_MODEL, LANDMARK_INDICES
from .config import TrackerConfig
from .types import HeadPose

try:
    from .tracker import OrbbecHeadTracker
except Exception:  # pragma: no cover
    OrbbecHeadTracker = None  # type: ignore[assignment]

__all__ = [
    "FACE_3D_MODEL",
    "LANDMARK_INDICES",
    "HeadPose",
    "OrbbecHeadTracker",
    "TrackerConfig",
]
