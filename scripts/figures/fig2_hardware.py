"""Figure 2 - System / hardware architecture of the adaptive-printing rig.

A publication-quality vector schematic of the physical setup: Orbbec Gemini 2L
depth camera observing the subject's head, a 5-axis CNC (X/Y/Z linear + B/C
rotary) carrying the print nozzle, and the HICON / Mach4 control chain. The
nozzle arm is annotated with the true link lengths from the machine
calibration (a_mm, d_mm).

Run:   python scripts/figures/fig2_hardware.py
       python scripts/figures/fig2_hardware.py --photo path/to/rig.jpg
Out:   results/figures/fig2_hardware.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

from figstyle import DOUBLE_COL_IN, PALETTE, ROLE, apply_style, draw_arrow, draw_box, save_figure

# Machine calibration (from config/cnc_compensation_example.yaml).
A_MM = 180.7   # C-arm structural length
D_MM = 57.59   # nozzle/tool length past the B pivot


def _struct(ax: plt.Axes) -> None:
    steel = PALETTE["grey"]
    light = "#D9D9D9"
    # Base / table.
    ax.add_patch(Rectangle((18, 6), 66, 5, facecolor=steel, edgecolor="black", lw=0.8, zorder=2))
    # Gantry columns.
    ax.add_patch(Rectangle((20, 11), 5, 70, facecolor=light, edgecolor="black", lw=0.8, zorder=2))
    ax.add_patch(Rectangle((77, 11), 5, 70, facecolor=light, edgecolor="black", lw=0.8, zorder=2))
    # Top X beam.
    ax.add_patch(Rectangle((20, 80), 62, 6, facecolor=light, edgecolor="black", lw=0.8, zorder=2))
    # Z carriage on the beam.
    ax.add_patch(Rectangle((46, 56), 12, 26, facecolor=light, edgecolor="black", lw=0.8, zorder=3))


def _arm_and_nozzle(ax: plt.Axes) -> None:
    """Stylised 5-axis wrist: C rotary -> a-arm -> B rotary -> d-tool -> nozzle."""
    machine_c = ROLE["machine"]
    # C rotary joint at the bottom of the Z carriage.
    c_joint = (52.0, 56.0)
    ax.add_patch(Circle(c_joint, 2.6, facecolor=machine_c, edgecolor="black", lw=0.8, zorder=5))
    # a-arm down to the B pivot (length ~ A_MM, scaled).
    b_pivot = (50.0, 45.0)
    ax.plot([c_joint[0], b_pivot[0]], [c_joint[1], b_pivot[1]],
            color=machine_c, lw=4.0, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle(b_pivot, 2.2, facecolor=machine_c, edgecolor="black", lw=0.8, zorder=5))
    # d-tool to the nozzle tip (length ~ D_MM, scaled), slightly tilted (B).
    tip = (47.5, 38.5)
    ax.plot([b_pivot[0], tip[0]], [b_pivot[1], tip[1]],
            color=machine_c, lw=3.0, solid_capstyle="round", zorder=4)
    ax.add_patch(Polygon([(tip[0] - 1.6, tip[1] + 1.2), (tip[0] + 1.6, tip[1] + 1.2), tip],
                         closed=True, facecolor=PALETTE["vermillion"], edgecolor="black",
                         lw=0.6, zorder=6))

    # C rotation axis (vertical, dashed) + arc.
    ax.plot([c_joint[0], c_joint[0]], [c_joint[1] - 1, 74], color="black",
            lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.add_patch(Arc((c_joint[0], 72), 12, 5, angle=0, theta1=200, theta2=340,
                     color="black", lw=0.9, zorder=3))
    ax.annotate("", xy=(c_joint[0] + 6, 71.2), xytext=(c_joint[0] + 5.2, 73.2),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9))
    ax.text(c_joint[0] + 7.5, 73, "C", fontsize=8, fontweight="bold")

    # B rotation arc at the pivot.
    ax.add_patch(Arc(b_pivot, 10, 10, angle=0, theta1=200, theta2=290,
                     color="black", lw=0.9, zorder=3))
    ax.text(b_pivot[0] + 5.5, b_pivot[1] - 3.5, "B", fontsize=8, fontweight="bold")

    # Link-length callouts.
    ax.annotate(f"a = {A_MM:.1f} mm", xy=(51, 50.5), xytext=(60, 52),
                fontsize=6.8, color=machine_c,
                arrowprops=dict(arrowstyle="-", color=machine_c, lw=0.7))
    ax.annotate(f"d = {D_MM:.1f} mm", xy=(48.8, 41.7), xytext=(60, 43),
                fontsize=6.8, color=machine_c,
                arrowprops=dict(arrowstyle="-", color=machine_c, lw=0.7))
    return tip


def _head(ax: plt.Axes, tip: tuple[float, float]) -> None:
    skin = PALETTE["purple"]
    center = (48.0, 26.0)
    ax.add_patch(Circle(center, 11.5, facecolor=skin, edgecolor="black", lw=0.9,
                        alpha=0.85, zorder=4))
    # Scalp trace under the nozzle.
    ax.add_patch(Arc(center, 23, 23, angle=0, theta1=55, theta2=125,
                     color="white", lw=2.2, zorder=5))
    ax.text(center[0], center[1] - 2, "subject\nhead", ha="center", va="center",
            color="white", fontsize=7.2, fontweight="bold", zorder=6)
    # Standoff between nozzle tip and scalp.
    ax.annotate("", xy=(tip[0], tip[1]), xytext=(tip[0], 37.3 - 0.2),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6))


def _camera(ax: plt.Axes) -> None:
    cam = ROLE["sensor"]
    body = FancyBboxPatch((3.5, 34), 9, 6, boxstyle="round,pad=0.0,rounding_size=0.6",
                          facecolor=cam, edgecolor="black", lw=0.8, zorder=5)
    ax.add_patch(body)
    ax.add_patch(Circle((11.5, 37), 1.4, facecolor="white", edgecolor="black", lw=0.7, zorder=6))
    ax.text(8, 41.5, "Orbbec\nGemini 2L", ha="center", va="bottom", fontsize=6.8,
            color=cam, fontweight="bold")
    # Camera stand.
    ax.add_patch(Rectangle((7, 8), 2, 26, facecolor=PALETTE["grey"], edgecolor="black",
                           lw=0.6, zorder=4))
    ax.add_patch(Rectangle((3, 6), 10, 2.5, facecolor=PALETTE["grey"], edgecolor="black",
                           lw=0.6, zorder=4))
    # Field of view onto the head.
    for dy in (-8, 8):
        ax.plot([12.5, 37], [37, 26 + dy], color=cam, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.text(22, 44, "RGB + depth\nfield of view", ha="center", fontsize=6.2,
            color=cam, style="italic")


def _control(ax: plt.Axes) -> None:
    vpc = draw_box(ax, (104, 66), "Vision PC\n(head tracking)", width=22, height=11,
                   facecolor=ROLE["vision"], textcolor="black", fontsize=7.0)
    hicon = draw_box(ax, (104, 44), "HICON\ncontroller", width=22, height=11,
                     facecolor=ROLE["machine"], textcolor="black", fontsize=7.0)
    mach4 = draw_box(ax, (104, 22), "Mach4 CNC PC\n(G-code + DRO)", width=22, height=11,
                     facecolor=ROLE["control"], textcolor="white", fontsize=7.0)

    # Camera -> Vision PC (USB3).
    draw_arrow(ax, (12.5, 39.5), (93, 66), connectionstyle="arc3,rad=0.28",
               color=PALETTE["grey"], lw=1.0)
    ax.text(88, 72, "USB3", fontsize=6.4, color=PALETTE["grey"], ha="center")
    # Vision PC -> HICON (UDP offsets).
    draw_arrow(ax, (104, 60.5), (104, 49.5), color=ROLE["safety"], lw=1.4)
    ax.text(106.5, 55, "UDP XYZBC\n@ 100 Hz", fontsize=6.2, color=ROLE["safety"], ha="left")
    # Mach4 -> Vision PC (work pose).
    draw_arrow(ax, (104, 27.5), (104, 60.5), color=ROLE["control"], lw=1.0,
               connectionstyle="arc3,rad=-0.55", linestyle=(0, (5, 3)))
    ax.text(118.5, 44, "UDP work pose", fontsize=6.2, color=ROLE["control"],
            ha="center", rotation=90)
    # HICON -> machine drives.
    draw_arrow(ax, (93, 44), (60, 70), color=ROLE["machine"], lw=1.0,
               connectionstyle="arc3,rad=-0.2")
    ax.text(78, 60, "5-axis drive", fontsize=6.2, color=ROLE["machine"], ha="center")


def _axes_triads(ax: plt.Axes) -> None:
    ox, oy = 70, 14
    ax.annotate("", xy=(ox + 8, oy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.annotate("", xy=(ox, oy + 8), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.add_patch(Circle((ox, oy), 1.0, facecolor="white", edgecolor="black", lw=0.8))
    ax.add_patch(Circle((ox, oy), 0.28, facecolor="black", edgecolor="black"))
    ax.text(ox + 9, oy, "X", fontsize=7.5, va="center", fontweight="bold")
    ax.text(ox, oy + 9, "Z", fontsize=7.5, ha="center", fontweight="bold")
    ax.text(ox - 2.4, oy - 2.4, "Y", fontsize=7.5, ha="center", fontweight="bold")
    ax.text(ox + 4, oy - 4, "machine frame", fontsize=6.2, ha="center", style="italic")

    # Linear-axis travel arrows.
    ax.annotate("", xy=(74, 89), xytext=(28, 89),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.9))
    ax.text(51, 90.5, "X travel", fontsize=6.6, ha="center")
    ax.annotate("", xy=(64, 80), xytext=(64, 58),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.9))
    ax.text(66, 69, "Z", fontsize=6.6, va="center")


def build(photo: str | None = None) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 4.2))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 96)
    ax.axis("off")
    ax.set_aspect("equal")

    if photo:
        img = plt.imread(photo)
        ax.imshow(img, extent=(0, 96, 0, 96), zorder=0, aspect="auto")
        ax.text(48, 93, "annotate components on the photo as needed",
                ha="center", fontsize=6.5, style="italic", color="white")
        _control(ax)
        fig.tight_layout(pad=0.4)
        return fig

    _struct(ax)
    tip = _arm_and_nozzle(ax)
    _head(ax, tip)
    _camera(ax)
    _control(ax)
    _axes_triads(ax)

    fig.tight_layout(pad=0.4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=str, default=None,
                        help="Optional rig photo to render instead of the schematic")
    args = parser.parse_args()
    fig = build(args.photo)
    for p in save_figure(fig, "fig2_hardware"):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
