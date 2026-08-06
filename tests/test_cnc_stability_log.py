from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.cnc_stability_log import (
    STABILITY_FIELDS,
    ChannelSpec,
    StabilityCsvLogger,
    channel_drift,
    linear_drift_per_hour,
    load_stability_csv,
    rolling_mean,
    summarize_drift,
    tracking_availability,
    valid_mask,
    write_drift_summary_csv,
)
from orbbec_head_tracking.types import HeadPose


def _pose(x: float = 1.0, y: float = 2.0, z: float = 600.0) -> HeadPose:
    return HeadPose(
        rotation_vector=np.array([[0.1], [0.2], [0.3]], dtype=np.float64),
        translation_vector_mm=np.array([x, y, z], dtype=np.float64),
        euler_degrees=(3.0, -4.0, 5.0),
        landmarks_2d=np.zeros((6, 2), dtype=np.float32),
        sampled_depth_mm=np.full(6, z, dtype=np.float32),
        inliers=None,
        solver="depth-rigid",
        valid_depth_count=6,
        reprojection_error_px=None,
        confidence=0.9,
    )


def _record(logger: StabilityCsvLogger, now: float, **overrides: object) -> bool:
    kwargs: dict[str, object] = dict(
        now_sec=now,
        elapsed_sec=now,
        pose=_pose(),
        head_delta_machine_mm=np.array([0.5, -0.25, 0.125]),
        required=CncUserOffset(1.0, 2.0, 3.0, 4.0, 5.0),
        sent=CncUserOffset(0.9, 1.9, 2.9, 3.9, 4.9),
        confidence=0.9,
        head_speed_mm_s=1.5,
        tracking_ok=True,
        baseline_ready=True,
        link_ok=True,
        safety_action="pass",
        safety_reason="",
    )
    kwargs.update(overrides)
    return logger.record(**kwargs)  # type: ignore[arg-type]


def test_logger_writes_header_and_row(tmp_path: Path) -> None:
    path = tmp_path / "stability.csv"
    with StabilityCsvLogger(path, rate_hz=10.0) as logger:
        assert _record(logger, 0.0) is True

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(STABILITY_FIELDS)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert float(row["head_z_mm"]) == 600.0
    assert float(row["pitch_deg"]) == 3.0
    assert float(row["head_dx_mm"]) == 0.5
    assert float(row["req_b"]) == 4.0
    assert float(row["off_b"]) == 3.9
    assert row["tracking_ok"] == "1"
    assert row["safety_action"] == "pass"
    assert row["wall_clock"] != ""


def test_logger_rate_limits_samples(tmp_path: Path) -> None:
    path = tmp_path / "stability.csv"
    with StabilityCsvLogger(path, rate_hz=10.0) as logger:
        written = [_record(logger, t / 100.0) for t in range(100)]

    assert sum(written) == 10
    assert logger.rows_written == 10


def test_logger_blanks_missing_pose_and_offsets(tmp_path: Path) -> None:
    path = tmp_path / "stability.csv"
    with StabilityCsvLogger(path, rate_hz=0.0) as logger:
        _record(
            logger,
            0.0,
            pose=None,
            head_delta_machine_mm=None,
            required=None,
            tracking_ok=False,
            baseline_ready=False,
            safety_action="hold_last",
            safety_reason="tracking_lost",
        )

    log = load_stability_csv(path)
    assert np.isnan(log["head_x_mm"][0])
    assert np.isnan(log["req_x"][0])
    assert log["off_x"][0] == 0.9
    assert log["tracking_ok"][0] == 0.0
    assert log["safety_reason"][0] == "tracking_lost"


def test_valid_mask_and_availability(tmp_path: Path) -> None:
    path = tmp_path / "stability.csv"
    with StabilityCsvLogger(path, rate_hz=0.0) as logger:
        _record(logger, 0.0)
        _record(logger, 1.0, tracking_ok=False, pose=None)
        _record(logger, 2.0, baseline_ready=False)

    log = load_stability_csv(path)
    assert valid_mask(log).tolist() == [True, False, False]
    assert tracking_availability(log) == 2.0 / 3.0


def test_linear_drift_per_hour_on_ramp() -> None:
    t = np.arange(0.0, 7200.0, 1.0)
    values = 0.25 * (t / 3600.0)  # 0.25 mm per hour
    assert linear_drift_per_hour(t, values) == pytest.approx(0.25)
    assert np.isnan(linear_drift_per_hour(t[:1], values[:1]))


def test_channel_drift_ignores_nan_and_uses_edge_windows() -> None:
    t = np.arange(0.0, 3600.0, 1.0)
    values = 0.5 * (t / 3600.0)
    values[100:200] = np.nan
    spec = ChannelSpec("off_x", "CNC offset X", "mm")
    stats = channel_drift(t, values, spec, edge_window_sec=60.0)

    assert stats.count == t.size - 100
    assert stats.slope_per_hour == pytest.approx(0.5)
    assert stats.drift_over_run == pytest.approx(0.5 * (3599.0 / 3600.0))
    assert stats.start_value < stats.end_value
    assert stats.peak_to_peak == pytest.approx(0.5 * (3599.0 / 3600.0))


def test_summarize_and_write_summary(tmp_path: Path) -> None:
    path = tmp_path / "stability.csv"
    with StabilityCsvLogger(path, rate_hz=0.0) as logger:
        for i in range(10):
            _record(logger, float(i))

    log = load_stability_csv(path)
    stats = summarize_drift(log)
    keys = [item.channel for item in stats]
    assert "off_x" in keys and "pitch_deg" in keys and "head_z_mm" in keys

    summary = write_drift_summary_csv(tmp_path / "summary.csv", stats)
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(stats)
    assert rows[0]["channel"] == "off_x"
    assert rows[0]["unit"] == "mm"


def test_rolling_mean_is_nan_aware() -> None:
    t = np.arange(0.0, 10.0, 1.0)
    values = np.ones(10)
    values[5] = np.nan
    smoothed = rolling_mean(t, values, window_sec=3.0)
    assert np.allclose(smoothed, 1.0)
    assert rolling_mean(t, values, window_sec=0.0) is values
