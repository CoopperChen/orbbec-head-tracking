from __future__ import annotations

import cv2
import numpy as np

from .cnc_offset_encoder import CncUserOffset
from .cnc_safety import SafetyDecision


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
) -> np.ndarray:
    out = canvas.copy()
    panel_width = 420
    panel_height = 168
    x0, y0 = 12, canvas.shape[0] - panel_height - 12
    x1, y1 = x0 + panel_width, y0 + panel_height
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 24, 32), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0.0, out)
    border_color = (80, 220, 120) if link_ok and decision.action == "pass" else (80, 120, 255)
    cv2.rectangle(out, (x0, y0), (x1, y1), border_color, 1)

    if baseline_capturing:
        baseline_text = "baseline: capturing..."
    elif baseline_ready:
        baseline_text = "baseline: ready"
    else:
        baseline_text = "baseline: waiting"

    rows = [
        f"CNC UDP  {link_label}   {baseline_text}",
        f"conf {confidence:4.2f}   safety {decision.action}"
        + (f" ({decision.reason})" if decision.reason else ""),
        (
            f"X {offset.x:7.2f}  Y {offset.y:7.2f}  Z {offset.z:7.2f} mm"
        ),
        (
            f"B {offset.b:7.2f}  C {offset.c:7.2f} deg"
        ),
    ]
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
