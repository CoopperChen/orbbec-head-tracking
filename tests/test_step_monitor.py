from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.stream_cnc import StepMonitor


def _offset(x: float = 0.0, y: float = 0.0, z: float = 0.0, b: float = 0.0, c: float = 0.0):
    return CncUserOffset(x, y, z, b, c)


def test_reports_translation_and_rotation_step() -> None:
    monitor = StepMonitor(1.0, 0.5)
    step_mm, step_deg = monitor.observe(_offset(), _offset(x=3.0, y=4.0, b=0.3), 0.0)
    assert step_mm == pytest.approx(5.0)
    assert step_deg == pytest.approx(0.3)


def test_tracks_running_maxima() -> None:
    monitor = StepMonitor(1.0, 0.5)
    monitor.observe(_offset(), _offset(x=0.4), 0.0)
    monitor.observe(_offset(x=0.4), _offset(x=2.4), 1.0)
    monitor.observe(_offset(x=2.4), _offset(x=2.5), 2.0)
    assert monitor.max_mm == pytest.approx(2.0)
    assert monitor.over_threshold == 1


def test_warnings_are_throttled(capsys: pytest.CaptureFixture[str]) -> None:
    monitor = StepMonitor(1.0, 0.5, cooldown_sec=1.0)
    for i in range(5):
        monitor.observe(_offset(), _offset(x=2.0), i * 0.1)
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1
    assert monitor.over_threshold == 5

    monitor.observe(_offset(), _offset(x=2.0), 5.0)
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_zero_threshold_disables_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    monitor = StepMonitor(0.0, 0.0)
    monitor.observe(_offset(), _offset(x=50.0, b=20.0), 0.0)
    assert capsys.readouterr().out == ""
    assert monitor.over_threshold == 0
    assert monitor.max_mm == pytest.approx(50.0)


def test_summary_mentions_both_units() -> None:
    monitor = StepMonitor(1.0, 0.5)
    monitor.observe(_offset(), _offset(x=2.0, b=1.0), 0.0)
    summary = monitor.summary()
    assert "2.00 mm" in summary and "1.00 deg" in summary
