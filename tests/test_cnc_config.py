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
    assert config.safety.recovery_ticks_after_hold == 20
    assert config.motor_map.x_motors == (0, 3)
    assert config.motor_map.b_motors == (4,)
    assert config.mismatch.enabled is True
    assert config.mismatch.kp == pytest.approx(0.0)
    assert config.mismatch.snap_enabled is False
    assert config.safety.on_tracking_loss == "hold_last"
    assert config.safety.on_low_confidence == "hold_last"
    assert config.safety.on_head_speed_exceeded == "hold_last"
    assert config.safety.head_speed_filter_alpha == pytest.approx(0.25)
    assert config.safety.head_speed_exceed_ticks == 3
    assert config.offset_deadband.enabled is True
    assert config.offset_deadband.enter_translation_mm == pytest.approx(0.6)
    assert config.bc_axis_sign.b == pytest.approx(-1.0)
    assert config.bc_axis_sign.c == pytest.approx(-1.0)
    assert config.bc_mode == "tool_normal_ik"
    assert config.bc_camera_rvec_mapping.b_axis == "z"
    assert config.bc_camera_rvec_mapping.c_axis == "y"
