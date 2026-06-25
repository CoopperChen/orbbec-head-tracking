from __future__ import annotations

import numpy as np
import pytest

from orbbec_head_tracking.cnc_kinematics import (
    MachineConfig,
    bc_from_normal,
    nozzle_tip,
    solve_xyz_for_tip_delta,
)


def test_nozzle_tip_at_machine_zero_pose() -> None:
    machine = MachineConfig()
    tip = nozzle_tip(0.0, 0.0, 0.0, 0.0, 90.0, machine)
    assert tip[0] == pytest.approx(0.0, abs=0.1)
    assert tip[1] == pytest.approx(-machine.a_mm, abs=0.1)
    assert tip[2] == pytest.approx(-machine.d_mm, abs=0.1)


def test_bc_from_normal_layout_design_fallback() -> None:
    from orbbec_head_tracking.cnc_kinematics import (
        bc_from_normal,
        bc_from_normal_layout_design,
        layout_design_available,
    )

    n = np.array([0.0, 0.0, 1.0])
    vendored = bc_from_normal(n)
    bridged = bc_from_normal_layout_design(n)
    if not layout_design_available():
        assert bridged == pytest.approx(vendored, abs=1e-6)


def test_solve_xyz_tip_delta_moves_tip() -> None:
    machine = MachineConfig()
    x0, y0, z0 = 0.0, 0.0, 0.0
    b0, c0 = 0.0, 90.0
    tip0 = nozzle_tip(x0, y0, z0, b0, c0, machine)
    x1, y1, z1 = solve_xyz_for_tip_delta(
        x0, y0, z0, b0, c0, np.array([1.0, 0.0, 0.0]), machine
    )
    tip1 = nozzle_tip(x1, y1, z1, b0, c0, machine)
    delta = tip1 - tip0
    assert np.linalg.norm(delta) == pytest.approx(1.0, abs=0.15)
