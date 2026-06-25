from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

@dataclass(frozen=True)
class MachineConfig:
    a_mm: float = 180.7
    d_mm: float = 57.59
    gap_size_mm: float = 15.0
    calgap_z_mm: float = 26.62
    c0_deg: float = 90.0
    b0_deg: float = 0.0


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_u = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis_u)
    if n < 1e-12:
        raise ValueError("rotation axis must be non-zero")
    axis_u = axis_u / n
    v = np.asarray(v, dtype=float)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(axis_u, v) * s + axis_u * np.dot(axis_u, v) * (1.0 - c)


def _tool_direction_from_arm(arm_vec: np.ndarray, b_deg: float) -> np.ndarray:
    na = np.linalg.norm(arm_vec)
    if na < 1e-12:
        raise ValueError("arm vector must be non-zero")
    arm_hat = arm_vec / na
    ref = np.array([0.0, 0.0, -1.0])
    tool_ref = ref - np.dot(ref, arm_hat) * arm_hat
    nt = np.linalg.norm(tool_ref)
    if nt < 1e-9:
        tool_ref = np.cross(arm_hat, np.array([1.0, 0.0, 0.0]))
        nt = np.linalg.norm(tool_ref)
    tool_ref = tool_ref / nt
    return _rodrigues(tool_ref, arm_hat, -np.deg2rad(float(b_deg)))


def structural_arm_offset(c_deg: float, a_mm: float) -> np.ndarray:
    c_rad = np.deg2rad(float(c_deg))
    arm_dir = rot_z(-c_rad) @ np.array([1.0, 0.0, 0.0])
    return float(a_mm) * arm_dir


def structural_arm_joints(
    center_xyz: np.ndarray,
    b_deg: float,
    c_deg: float,
    a_mm: float,
    d_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(center_xyz, dtype=float).reshape(3)
    arm_vec = structural_arm_offset(c_deg, a_mm)
    tool_dir = _tool_direction_from_arm(arm_vec, b_deg)
    b_pivot = center + arm_vec
    tip = b_pivot + float(d_mm) * tool_dir
    return center, b_pivot, tip


def nozzle_tip(
    x: float,
    y: float,
    z: float,
    b_deg: float,
    c_deg: float,
    machine: MachineConfig,
) -> np.ndarray:
    _c, _b, tip = structural_arm_joints(
        np.array([x, y, z], dtype=float),
        b_deg,
        c_deg,
        machine.a_mm,
        machine.d_mm,
    )
    return tip


def tool_normal_from_bc(b_deg: float, c_deg: float, machine: MachineConfig) -> np.ndarray:
    _c, b_pivot, tip = structural_arm_joints(
        np.zeros(3), b_deg, c_deg, machine.a_mm, machine.d_mm
    )
    arm = b_pivot - _c
    tool = tip - b_pivot
    n = tool / max(np.linalg.norm(tool), 1e-12)
    return n


def find_angle(comp_v: np.ndarray, proj_v: np.ndarray) -> float:
    denom = np.linalg.norm(comp_v) * np.linalg.norm(proj_v)
    if denom == 0:
        return float("nan")
    cos_angle = float(np.clip(np.dot(proj_v, comp_v) / denom, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_angle)))


def find_caxis_angle(normal: np.ndarray) -> float:
    norm_v = np.asarray(normal, dtype=float).reshape(1, 3)
    c_vc = np.array([1.0, 0.0, 0.0])
    p_vc = np.array([norm_v[0, 0], norm_v[0, 1], 0.0])
    sign_angle = np.cross(p_vc, c_vc)
    if sign_angle[2] == 0:
        return 0.0
    return float(np.sign(sign_angle[2]) * (-90 + find_angle(c_vc, p_vc)))


def find_baxis_angle(normal: np.ndarray) -> float:
    norm_v = np.asarray(normal, dtype=float).reshape(3)
    c_vb = norm_v
    p_vb = np.array([norm_v[0], norm_v[1], 0.0])
    if c_vb[2] >= 0:
        return float(np.sign(norm_v[1]) * (find_angle(-c_vb, p_vb) - 90))
    return float(np.sign(norm_v[1]) * (270 - find_angle(-c_vb, p_vb)))


def bc_from_normal(normal: np.ndarray) -> tuple[float, float]:
    n = np.asarray(normal, dtype=float).reshape(3)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return 0.0, 0.0
    n = n / norm
    b = find_baxis_angle(n)
    c = find_caxis_angle(n)
    if np.isnan(b):
        b = 0.0
    if np.isnan(c):
        c = 0.0
    return float(b), float(c)


def rvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    import cv2

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rmat, _ = cv2.Rodrigues(rvec)
    return np.asarray(rmat, dtype=np.float64)


def matrix_to_rvec(rmat: np.ndarray) -> np.ndarray:
    import cv2

    rvec, _ = cv2.Rodrigues(np.asarray(rmat, dtype=np.float64))
    return np.asarray(rvec, dtype=np.float64).reshape(3)


def rotation_delta_matrix(r_ref: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    return np.asarray(r_ref, dtype=np.float64).T @ np.asarray(r_current, dtype=np.float64)


def numerical_jacobian_xyz(
    x: float,
    y: float,
    z: float,
    b_deg: float,
    c_deg: float,
    machine: MachineConfig,
    eps: float = 0.05,
) -> np.ndarray:
    base = nozzle_tip(x, y, z, b_deg, c_deg, machine)
    jac = np.zeros((3, 3), dtype=float)
    for i, delta in enumerate((eps, 0.0, 0.0)):
        p = [x, y, z]
        p[i] += delta
        jac[:, i] = (nozzle_tip(p[0], p[1], p[2], b_deg, c_deg, machine) - base) / eps
    return jac


def solve_xyz_for_tip_delta(
    x: float,
    y: float,
    z: float,
    b_deg: float,
    c_deg: float,
    tip_delta: np.ndarray,
    machine: MachineConfig,
    *,
    max_iter: int = 12,
    tol_mm: float = 0.05,
) -> tuple[float, float, float]:
    px, py, pz = float(x), float(y), float(z)
    target = nozzle_tip(px, py, pz, b_deg, c_deg, machine) + np.asarray(tip_delta, dtype=float).reshape(3)
    for _ in range(max_iter):
        tip = nozzle_tip(px, py, pz, b_deg, c_deg, machine)
        err = target - tip
        if float(np.linalg.norm(err)) < tol_mm:
            break
        jac = numerical_jacobian_xyz(px, py, pz, b_deg, c_deg, machine)
        step, *_ = np.linalg.lstsq(jac, err, rcond=None)
        px += float(step[0])
        py += float(step[1])
        pz += float(step[2])
    return px, py, pz


def default_head_normal() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0], dtype=float)


def normal_from_pose_rotation(rmat: np.ndarray) -> np.ndarray:
    n = np.asarray(rmat, dtype=float) @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return default_head_normal()
    return n / norm


def layout_design_available() -> bool:
    """Return True when layout_design is importable (LAYOUT_DESIGN_ROOT or PYTHONPATH)."""
    return _load_layout_design() is not None


def _load_layout_design() -> tuple[Any, Any] | None:
    import os
    import sys

    root = os.environ.get("LAYOUT_DESIGN_ROOT")
    if root:
        app_parent = str(Path(root).resolve())
        if app_parent not in sys.path:
            sys.path.insert(0, app_parent)
    try:
        from app.postprocess.gcode.kinematics.axis_angles import compute_axis_angles
        from app.postprocess.gcode.kinematics.machine_fk import structural_arm_joints as ld_joints
    except ImportError:
        return None
    return compute_axis_angles, ld_joints


def bc_from_normal_layout_design(normal: np.ndarray) -> tuple[float, float]:
    """B/C from layout_design when available; falls back to vendored implementation."""
    loaded = _load_layout_design()
    if loaded is None:
        return bc_from_normal(normal)
    compute_axis_angles, _ = loaded
    n = np.asarray(normal, dtype=float).reshape(1, 3)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return 0.0, 0.0
    n = n / norm
    b_angles, c_angles = compute_axis_angles(n)
    return float(b_angles[0]), float(c_angles[0])
