from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_config import CncCompensationConfig, OffsetDeadbandConfig
from orbbec_head_tracking.cnc_mismatch import CncMismatchTracker
from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder, CncUserOffset
from orbbec_head_tracking.cnc_safety import CncSafetyGuards
from orbbec_head_tracking.cnc_offset_encoder import CncOffsetEncoder, CncUserOffset, OffsetZeroLatch
from orbbec_head_tracking.types import HeadPose


def _make_pose(
    t_mm: tuple[float, float, float],
    rvec: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> HeadPose:
    r = np.array(rvec, dtype=np.float32).reshape(3, 1)
    return HeadPose(
        rotation_vector=r,
        translation_vector_mm=np.array(t_mm, dtype=np.float32),
        euler_degrees=(0.0, 0.0, 0.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.zeros(6, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=1.0,
        confidence=1.0,
        smoothed=False,
    )


def test_zero_latch_hysteresis() -> None:
    latch = OffsetZeroLatch(
        OffsetDeadbandConfig(
            enter_translation_mm=0.5,
            exit_translation_mm=1.0,
            enter_rotation_deg=0.4,
            exit_rotation_deg=0.8,
        )
    )
    assert latch.apply(CncUserOffset(0.3, 0.0, 0.0, 0.0, 0.0)) == CncUserOffset.zero()
    assert latch.apply(CncUserOffset(1.5, 0.0, 0.0, 0.0, 0.0)).x == pytest.approx(1.5)
    assert latch.apply(CncUserOffset(0.4, 0.0, 0.0, 0.0, 0.0)) == CncUserOffset.zero()
    assert latch.apply(CncUserOffset(0.3, 0.0, 0.0, 0.0, 0.0)) == CncUserOffset.zero()


def test_zero_latch_releases_diagonal_translation() -> None:
    latch = OffsetZeroLatch(
        OffsetDeadbandConfig(
            enter_translation_mm=0.4,
            exit_translation_mm=1.0,
        )
    )
    released = latch.apply(CncUserOffset(0.8, 0.6, 0.5, 0.0, 0.0))
    assert released.x == pytest.approx(0.8)
    assert released.y == pytest.approx(0.6)
    assert released.z == pytest.approx(0.5)


def test_encoder_zero_latch_suppresses_pose_noise() -> None:
    config = CncCompensationConfig(
        offset_deadband=OffsetDeadbandConfig(
            enter_translation_mm=0.5,
            exit_translation_mm=1.0,
        ),
    )
    encoder = CncOffsetEncoder(config)
    encoder.capture_baseline(_make_pose((0.0, 0.0, 500.0)))
    noisy = _make_pose((0.3, -0.2, 500.1))
    offset = encoder.encode(noisy)
    assert offset == CncUserOffset.zero()


def test_mismatch_targets_zero_without_kp_boost() -> None:
    config = CncCompensationConfig()
    tracker = CncMismatchTracker(config)
    required = CncUserOffset.zero()
    sent = CncUserOffset(4.0, 0.0, 0.0, 0.0, 0.0)
    target, report = tracker.target(required, sent, dt_sec=0.01, head_speed_mm_s=0.0)
    assert target == CncUserOffset.zero()
    assert report.mode == "zero"


def test_safety_zero_settle_snaps_small_residual() -> None:
    config = CncCompensationConfig(
        offset_deadband=OffsetDeadbandConfig(
            exit_translation_mm=1.2,
            exit_rotation_deg=1.0,
        ),
    )
    safety = CncSafetyGuards(config)
    safety._state.last_offset = CncUserOffset(0.4, 0.2, 0.0, 0.0, 0.0)
    decision = safety.evaluate(
        CncUserOffset.zero(),
        spike_reference=CncUserOffset.zero(),
        tracking_ok=True,
        confidence=1.0,
        baseline_ready=True,
        head_speed_mm_s=0.0,
    )
    assert decision.offset == CncUserOffset.zero()
    assert decision.reason == ""


def test_safety_zero_ramp_drains_large_residual() -> None:
    config = CncCompensationConfig(update_period_ms=10.0)
    safety = CncSafetyGuards(config)
    safety._state.last_offset = CncUserOffset(5.0, 0.0, 0.0, 0.0, 0.0)
    last = safety._state.last_offset
    for _ in range(300):
        last = safety.evaluate(
            CncUserOffset.zero(),
            spike_reference=CncUserOffset.zero(),
            tracking_ok=True,
            confidence=1.0,
            baseline_ready=True,
            head_speed_mm_s=0.0,
        ).offset
    assert last.x == pytest.approx(0.0, abs=0.05)
