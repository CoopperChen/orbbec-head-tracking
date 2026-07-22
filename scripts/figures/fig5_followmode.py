"""Figure 5 - Follow-mode motion compensation.

(a) Concept: when the head moves by delta (X/Y/Z + pitch/yaw/roll), the encoder
    commands an equal XYZBC nozzle offset so the tool tip stays locked to the
    scalp. (b) Time series from a real/synthetic session log: commanded nozzle
    offset tracking the head displacement.

First produce a log with log_follow_mode.py, then:
  python scripts/figures/fig5_followmode.py
Out: results/figures/fig5_followmode.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon

from figstyle import DOUBLE_COL_IN, PALETTE, ROLE, apply_style, save_figure

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO / "results" / "follow_mode_log.csv"


def _nozzle(ax: plt.Axes, tip: tuple[float, float], color: str, alpha: float = 1.0,
            ls: str = "-") -> None:
    carriage_y = tip[1] + 3.6
    ax.plot([tip[0], tip[0]], [carriage_y, tip[1] + 0.5], color=color, lw=2.6,
            alpha=alpha, ls=ls, solid_capstyle="round", zorder=5)
    ax.add_patch(Polygon([(tip[0] - 0.5, tip[1] + 0.6), (tip[0] + 0.5, tip[1] + 0.6), tip],
                         closed=True, facecolor=color, edgecolor="black", lw=0.5,
                         alpha=alpha, zorder=6))


def _panel_a(ax: plt.Axes) -> None:
    ax.set_title("(a) Follow-mode compensation", fontsize=8.5)
    ax.set_xlim(-4.5, 10.5)
    ax.set_ylim(-5.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    c0 = np.array([0.0, 0.0])
    delta = np.array([5.0, 2.4])
    c1 = c0 + delta
    r = 2.0
    p0 = c0 + np.array([0.0, r])
    p1 = c1 + np.array([0.0, r])

    # Baseline head + nozzle (faint, dashed).
    ax.add_patch(Circle(c0, r, facecolor="none", edgecolor=PALETTE["grey"], lw=1.2,
                        ls=(0, (4, 3)), zorder=3))
    _nozzle(ax, tuple(p0), PALETTE["grey"], alpha=0.7, ls=(0, (3, 2)))
    ax.text(c0[0], c0[1] - r - 0.9, "baseline", ha="center", va="top",
            fontsize=6.8, color=PALETTE["grey"])

    # Displaced head + compensated nozzle (solid).
    ax.add_patch(Circle(c1, r, facecolor=ROLE["body"], edgecolor="black", lw=0.9,
                        alpha=0.85, zorder=4))
    _nozzle(ax, tuple(p1), ROLE["machine"])
    ax.text(c1[0], c1[1], "moved\nhead", ha="center", va="center",
            fontsize=6.6, color="white", fontweight="bold", zorder=7)

    # Head displacement arrow (between the two centres).
    ax.annotate("", xy=tuple(c1), xytext=tuple(c0),
                arrowprops=dict(arrowstyle="-|>", color=ROLE["body"], lw=1.8))
    ax.text(c0[0] - 0.4, c0[1] + 1.1, "head $\\Delta$\n(X,Y,Z + P/Y/R)",
            ha="right", va="center", fontsize=6.8, color=ROLE["body"])

    # Equal, parallel nozzle-offset arrow along the top.
    ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                arrowprops=dict(arrowstyle="-|>", color=ROLE["machine"], lw=1.8))
    ax.text((p0[0] + p1[0]) / 2, max(p0[1], p1[1]) + 1.4,
            "commanded XYZBC offset", ha="center", va="bottom",
            fontsize=6.8, color=ROLE["machine"])
    ax.scatter([p1[0]], [p1[1]], s=20, color=PALETTE["vermillion"], zorder=8)
    ax.annotate("tip stays\non scalp", xy=tuple(p1), xytext=(p1[0] + 2.6, p1[1] + 0.4),
                ha="left", va="center", fontsize=6.5, color=PALETTE["vermillion"],
                style="italic",
                arrowprops=dict(arrowstyle="-", color=PALETTE["vermillion"], lw=0.6))


def _read_log(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    cols: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for name in reader.fieldnames or []:
            cols[name] = []
        for row in reader:
            for name in cols:
                cols[name].append(float(row[name]))
    if not cols.get("t_s"):
        return None
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _panel_b(ax_top: plt.Axes, ax_bot: plt.Axes, log: dict[str, np.ndarray] | None) -> None:
    if log is None:
        for ax in (ax_top, ax_bot):
            ax.axis("off")
        ax_top.text(0.5, 0.5,
                    "No log found.\nRun scripts/figures/log_follow_mode.py\n"
                    "(--source synthetic or live)",
                    ha="center", va="center", fontsize=7.5, color=PALETTE["grey"],
                    transform=ax_top.transAxes)
        return

    t = log["t_s"]
    # Translation: commanded offset (solid) vs head delta (dashed).
    ax_top.set_title("(b) Nozzle offset tracks head motion", fontsize=8.5)
    for key_off, key_d, color, label in (
        ("off_x", "head_dx_mm", PALETTE["vermillion"], "X"),
        ("off_y", "head_dy_mm", PALETTE["green"], "Y"),
        ("off_z", "head_dz_mm", PALETTE["blue"], "Z"),
    ):
        ax_top.plot(t, log[key_off], color=color, lw=1.4, label=f"offset {label}")
        if key_d in log:
            ax_top.plot(t, log[key_d], color=color, lw=0.9, ls=(0, (3, 2)), alpha=0.6)
    ax_top.set_ylabel("offset (mm)")
    ax_top.legend(ncol=3, loc="upper right", frameon=False, fontsize=6.5,
                  handlelength=1.4, columnspacing=1.0)
    ax_top.grid(True, lw=0.4, alpha=0.4)
    ax_top.tick_params(labelbottom=False)

    ax_bot.plot(t, log["off_b"], color=PALETTE["orange"], lw=1.4, label="B")
    ax_bot.plot(t, log["off_c"], color=PALETTE["purple"], lw=1.4, label="C")
    ax_bot.set_ylabel("offset (deg)")
    ax_bot.set_xlabel("time (s)")
    ax_bot.legend(ncol=2, loc="upper right", frameon=False, fontsize=6.5, handlelength=1.4)
    ax_bot.grid(True, lw=0.4, alpha=0.4)
    ax_top.text(0.01, 0.02, "dashed = head $\\Delta$ (machine frame)", transform=ax_top.transAxes,
                fontsize=6.0, color=PALETTE["grey"], style="italic")


def build(log_path: Path) -> plt.Figure:
    apply_style()
    fig = plt.figure(figsize=(DOUBLE_COL_IN, 3.1), constrained_layout=True)
    gs = GridSpec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1, 1], figure=fig)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_bt = fig.add_subplot(gs[0, 1])
    ax_bb = fig.add_subplot(gs[1, 1])
    _panel_a(ax_a)
    _panel_b(ax_bt, ax_bb, _read_log(log_path))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=str, default=str(_DEFAULT_LOG))
    args = parser.parse_args()
    fig = build(Path(args.log))
    for p in save_figure(fig, "fig5_followmode"):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
