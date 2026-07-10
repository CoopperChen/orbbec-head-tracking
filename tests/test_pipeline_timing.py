from __future__ import annotations

import csv
from pathlib import Path

from orbbec_head_tracking.pipeline_timing import (
    LoopTimingSample,
    format_summary_table,
    summarize_timing,
    write_timing_csv,
)


def _sample(loop_index: int, **overrides: float | bool) -> LoopTimingSample:
    defaults = dict(
        loop_index=loop_index,
        wall_time_sec=float(loop_index),
        loop_period_ms=33.0,
        frame_acquire_align_ms=12.0,
        landmark_ms=8.0,
        pose_estimate_ms=3.0,
        temporal_filter_ms=1.0,
        encode_ms=0.2,
        safety_rate_limit_ms=0.1,
        udp_ms=0.05,
        loop_total_ms=24.5,
        tracking_ok=True,
        baseline_ready=True,
        face_detected=True,
    )
    defaults.update(overrides)
    return LoopTimingSample(**defaults)  # type: ignore[arg-type]


def test_write_timing_csv_roundtrip(tmp_path: Path) -> None:
    samples = [
        _sample(0),
        _sample(1, loop_total_ms=26.0, landmark_ms=9.0),
    ]
    path = tmp_path / "timing.csv"
    write_timing_csv(path, samples)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["loop_index"] == "0"
    assert float(rows[1]["landmark_ms"]) == 9.0
    assert rows[1]["tracking_ok"] == "True"


def test_summarize_timing_computes_percentiles() -> None:
    samples = [
        _sample(0, loop_total_ms=20.0),
        _sample(1, loop_total_ms=30.0),
        _sample(2, loop_total_ms=40.0),
        _sample(3, loop_total_ms=50.0, tracking_ok=False),
    ]
    summary = summarize_timing(samples)

    assert summary["loop_total_ms"]["median"] == 35.0
    assert summary["loop_total_ms"]["min"] == 20.0
    assert summary["loop_total_ms"]["max"] == 50.0
    assert summary["tracking_ok_rate"]["value"] == 0.75

    table = format_summary_table(summary)
    assert "loop_total_ms,35.000,35.000" in table
    assert "tracking_ok_rate,0.7500" in table


def test_summarize_timing_empty() -> None:
    assert summarize_timing([]) == {}
