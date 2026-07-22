"""Shared publication style for NSF on-body fabrication figures.

Import this from the ``figN_*.py`` scripts. It configures a colorblind-safe
palette (Okabe-Ito), legible serif typography, and a ``save_figure`` helper that
exports vector PDF + SVG alongside a high-DPI PNG into ``results/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Okabe-Ito colorblind-safe qualitative palette.
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
    "light": "#EDEDED",
}

# Logical role -> colour, so figures stay consistent with each other.
ROLE = {
    "sensor": PALETTE["blue"],
    "vision": PALETTE["sky"],
    "control": PALETTE["green"],
    "safety": PALETTE["vermillion"],
    "machine": PALETTE["orange"],
    "body": PALETTE["purple"],
    "io": PALETTE["grey"],
}

# Column widths (inches) for a two-column NSF/IEEE style layout.
SINGLE_COL_IN = 3.4
DOUBLE_COL_IN = 7.0

_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def apply_style() -> None:
    """Set global matplotlib rcParams for consistent, legible figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed as TrueType (editable, no Type-3 warnings)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, name: str, *, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> list[Path]:
    """Save ``fig`` to results/figures/<name>.<ext> for each format. Returns paths."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        out = _RESULTS_DIR / f"{name}.{ext}"
        fig.savefig(out)
        written.append(out)
    return written


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = "black",
    fontsize: float = 8.0,
    rounding: float = 0.06,
    alpha: float = 1.0,
    fontweight: str = "normal",
    zorder: float = 2.0,
) -> tuple[float, float]:
    """Draw a rounded, centred label box. ``xy`` is the box centre. Returns centre."""
    cx, cy = xy
    box = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=zorder + 1,
        wrap=True,
    )
    return cx, cy


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#000000",
    style: str = "-|>",
    lw: float = 1.3,
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
    mutation_scale: float = 12.0,
    zorder: float = 1.5,
) -> None:
    """Draw a directed connector between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        color=color,
        lw=lw,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=mutation_scale,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
