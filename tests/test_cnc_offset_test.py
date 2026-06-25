from __future__ import annotations

import pytest

from orbbec_head_tracking.cnc_offset_test import (
    _parse_limits,
    _trackbar_to_value,
    _value_to_trackbar,
)


def test_parse_limits() -> None:
    assert _parse_limits("-1,1") == (-1.0, 1.0)


def test_trackbar_roundtrip() -> None:
    minimum, maximum = -10.0, 10.0
    for value in (-10.0, 0.0, 2.5, 10.0):
        position = _value_to_trackbar(value, minimum, maximum)
        restored = _trackbar_to_value(position, minimum, maximum)
        assert restored == pytest.approx(value, abs=0.02)
