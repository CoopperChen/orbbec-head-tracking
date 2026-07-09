from __future__ import annotations

import cv2
import numpy as np

from .cnc_config import MachinePose
from .cnc_mismatch import MismatchReport
from .cnc_offset_encoder import CncUserOffset
from .cnc_safety import SafetyAction, SafetyDecision

_BENIGN_SAFETY_REASONS = frozenset({"zero_settled", "zero_ramp"})


def _safety_status_text(action: SafetyAction, reason: str, offset: CncUserOffset) -> str:
    if reason and reason not in _BENIGN_SAFETY_REASONS:
        return f"safety {action} ({reason})"
    if action == "hold_last":
        return f"safety {action} ({reason})" if reason else f"safety {action}"
    if action == "zero":
        return f"safety {action} ({reason})" if reason else f"safety {action}"
    if offset == CncUserOffset.zero():
        return "safety pass (idle @ zero)"
    return "safety pass (tracking)"


def draw_cnc_status_panel(
    canvas: np.ndarray,
    *,
    baseline_ready: bool,
    baseline_capturing: bool,
    offset: CncUserOffset,
    decision: SafetyDecision,
    link_ok: bool,
    link_label: str,
    confidence: float,
    required: CncUserOffset | None = None,
    mismatch_report: MismatchReport | None = None,
    work_pose_status: str | None = None,
    tool_pose: MachinePose | None = None,
) -> np.ndarray:
    out = canvas.copy()
    panel_width = 420
    panel_height = 220 if mismatch_report is not None else 168
    if work_pose_status is not None:
        panel_height += 26
    if tool_pose is not None:
        panel_height += 52
    x0, y0 = 12, canvas.shape[0] - panel_height - 12
    x1, y1 = x0 + panel_width, y0 + panel_height
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0.0, out)
    border_color = (80, 220, 120) if link_ok and decision.action == "pass" else (80, 120, 255)
    if mismatch_report is not None and mismatch_report.fault:
        border_color = (60, 80, 255)
    cv2.rectangle(out, (x0, y0), (x1, y1), border_color, 1)

    if baseline_capturing:
        baseline_text = "baseline: capturing..."
    elif baseline_ready:
        baseline_text = "baseline: ready"
    else:
        baseline_text = "baseline: waiting"

    rows = [
        f"CNC UDP  {link_label}   {baseline_text}",
        f"conf {confidence:4.2f}   {_safety_status_text(decision.action, decision.reason, offset)}",
        (
            f"sent X {offset.x:7.2f}  Y {offset.y:7.2f}  Z {offset.z:7.2f} mm"
        ),
        (
            f"sent B {offset.b:7.2f}  C {offset.c:7.2f} deg"
        ),
    ]
    if required is not None and mismatch_report is not None:
        rows.append(
            f"req  X {required.x:7.2f}  Y {required.y:7.2f}  Z {required.z:7.2f} mm"
        )
        mode = mismatch_report.mode
        rows.append(
            f"err {mismatch_report.error_norm_mm:6.2f} mm  "
            f"{mismatch_report.error_norm_deg:5.2f} deg  {mode}"
        )
    if work_pose_status is not None:
        rows.append(work_pose_status)
    if tool_pose is not None:
        rows.append(
            f"work X {tool_pose.x:7.2f}  Y {tool_pose.y:7.2f}  Z {tool_pose.z:7.2f} mm"
        )
        rows.append(
            f"work B {tool_pose.b_deg:7.2f}  C {tool_pose.c_deg:7.2f} deg"
        )
    cv2.putText(
        out,
        "CNC offset stream",
        (x0 + 16, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 245, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    for i, text in enumerate(rows):
        cv2.putText(
            out,
            text,
            (x0 + 16, y0 + 58 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 236, 244),
            1,
            lineType=cv2.LINE_AA,
        )
    return out
