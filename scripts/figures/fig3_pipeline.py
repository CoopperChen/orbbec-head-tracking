"""Figure 3 - Closed-loop adaptive-printing control pipeline.

Redraws the production ``orbbec-head-stream-cnc`` data flow (docs/cnc-udp-pipeline.md)
as a true closed loop: perception -> real-time control @100 Hz -> 5-axis CNC ->
nozzle held on the moving scalp -> body motion re-observed by the camera.

Run:  python scripts/figures/fig3_pipeline.py
Out:  results/figures/fig3_pipeline.{pdf,svg,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from figstyle import DOUBLE_COL_IN, ROLE, apply_style, draw_arrow, draw_box, save_figure

W, H = 22.0, 12.0  # box width / height in data units
DARK_TEXT = {"sensor", "vision", "safety", "control", "machine", "body", "io"}


def _text_color(role: str) -> str:
    # Lighter fills read better with black text.
    return "black" if role in {"vision", "machine"} else "white"


def build() -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 3.7))
    ax.set_xlim(-26, 122)
    ax.set_ylim(-4, 62)
    ax.axis("off")
    ax.set_aspect("equal")

    row1_y, row2_y = 50.0, 24.0
    xs = [12, 36, 60, 84, 108]

    perception = [
        ("Orbbec Gemini 2L\n(RGB + depth)", "sensor"),
        ("Depth-to-color\nalign", "vision"),
        ("MediaPipe\nFaceMesh", "vision"),
        ("6-DoF pose\n(depth-rigid)", "vision"),
        ("Temporal\nsmoothing", "vision"),
    ]
    control = [
        ("Encode XYZBC\nvs. baseline", "control"),
        ("Mismatch +\nrate limit", "control"),
        ("Safety guard\nhold-last on loss", "safety"),
        ("UDP XYZBC\n@ 100 Hz", "io"),
        ("HICON\ncontroller", "machine"),
    ]

    # Row 1: perception, left -> right.
    for x, (label, role) in zip(xs, perception):
        draw_box(ax, (x, row1_y), label, width=W, height=H,
                 facecolor=ROLE[role], textcolor=_text_color(role), fontsize=7.2)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        draw_arrow(ax, (x0 + W / 2, row1_y), (x1 - W / 2, row1_y))

    # Drop from smoothing (top-right) to control row (bottom-right).
    draw_arrow(ax, (xs[-1], row1_y - H / 2), (xs[-1], row2_y + H / 2))

    # Row 2: control, right -> left.
    xs_r = list(reversed(xs))
    for x, (label, role) in zip(xs_r, control):
        draw_box(ax, (x, row2_y), label, width=W, height=H,
                 facecolor=ROLE[role], textcolor=_text_color(role), fontsize=7.2)
    for x0, x1 in zip(xs_r[:-1], xs_r[1:]):
        draw_arrow(ax, (x0 - W / 2, row2_y), (x1 + W / 2, row2_y))

    # 5-axis CNC / nozzle-on-scalp node below the HICON controller (leftmost row-2 box).
    cnc_y = 4.0
    draw_box(ax, (12, cnc_y), "5-axis CNC: nozzle\nlocked to scalp", width=W, height=H,
             facecolor=ROLE["body"], textcolor="white", fontsize=7.2)
    draw_arrow(ax, (12, row2_y - H / 2), (12, cnc_y + H / 2))

    # Physical feedback loop: body moves -> camera re-observes.
    draw_arrow(ax, (12 - W / 2, cnc_y), (12 - W / 2, row1_y),
               color=ROLE["body"], linestyle=(0, (5, 3)),
               connectionstyle="arc3,rad=-0.55", lw=1.5)
    ax.text(-22.5, 27, "Physical closed loop:\nbody moves, camera re-observes",
            ha="center", va="center", rotation=90, fontsize=7.0,
            color=ROLE["body"], style="italic")

    # Group labels.
    ax.text(60, row1_y + H / 2 + 4.5, "Perception", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=ROLE["vision"])
    ax.text(60, row2_y + H / 2 + 4.5, "Real-time control loop @ 100 Hz", ha="center",
            va="bottom", fontsize=8.5, fontweight="bold", color=ROLE["control"])

    # Loss/spike annotation into the safety box.
    safety_x = xs_r[2]
    ax.text(safety_x, row2_y - H / 2 - 3.0, "tracking loss / spike\n-> no zero-flash",
            ha="center", va="top", fontsize=6.6, color=ROLE["safety"], style="italic")

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    fig = build()
    paths = save_figure(fig, "fig3_pipeline")
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
