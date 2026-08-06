"""Shutdown must never command motion, and must not let an offset go unnoticed."""

from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_offset_encoder import CncUserOffset
from orbbec_head_tracking.stream_cnc import EXIT_OFFSET_WARN_MM, _report_standing_offset


def test_zero_offset_says_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    _report_standing_offset(CncUserOffset.zero())
    assert capsys.readouterr().out == ""


def test_small_offset_gets_a_one_line_note(capsys: pytest.CaptureFixture[str]) -> None:
    _report_standing_offset(CncUserOffset(1.0, 0.5, 0.0, 0.0, 0.0))
    out = capsys.readouterr().out
    assert "!!" not in out
    assert "1.12 mm" in out
    assert "Mach4" in out


def test_large_offset_gets_a_banner_naming_the_jump(capsys: pytest.CaptureFixture[str]) -> None:
    """The run we analysed ended holding 20.3 mm under tracking_lost."""
    _report_standing_offset(CncUserOffset(-12.5693, -13.4857, 8.5114, -2.3844, -2.0))
    out = capsys.readouterr().out
    assert "!!" in out
    assert "20.3" in out
    assert "NOT moved" in out
    # The startup jump is the whole reason holding is only safe if the operator acts.
    assert "servo cycle" in out


def test_banner_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    _report_standing_offset(CncUserOffset(EXIT_OFFSET_WARN_MM - 0.01, 0.0, 0.0, 0.0, 0.0))
    assert "!!" not in capsys.readouterr().out
    _report_standing_offset(CncUserOffset(EXIT_OFFSET_WARN_MM + 0.01, 0.0, 0.0, 0.0, 0.0))
    assert "!!" in capsys.readouterr().out


def test_rotation_only_offset_is_still_reported(capsys: pytest.CaptureFixture[str]) -> None:
    _report_standing_offset(CncUserOffset(0.0, 0.0, 0.0, 3.0, -2.0))
    out = capsys.readouterr().out
    assert "B+3.000" in out and "C-2.000" in out


def test_exports_no_ramp_helper() -> None:
    import orbbec_head_tracking.cnc_safety as safety

    assert not hasattr(safety, "offset_ramp_to_zero")
