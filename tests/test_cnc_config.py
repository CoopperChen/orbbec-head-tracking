from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from orbbec_head_tracking.cnc_config import load_compensation_config


def test_load_example_calibration_yaml() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "cnc_compensation_example.yaml"
    config = load_compensation_config(path)
    assert config.machine.a_mm == pytest.approx(180.7)
    assert config.offset_mode == "follow"
    assert config.safety.min_confidence == pytest.approx(0.6)
    assert config.machine_pose is not None
    assert config.machine_pose.b_deg == pytest.approx(0.0)
    assert config.motor_map.x_motors == (0, 3)
    assert config.motor_map.b_motors == (4,)
