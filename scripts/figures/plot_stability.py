"""Drift plot for a long-run CNC stability session.

Reads a stability CSV written by ``orbbec-head-stream-cnc --log`` and plots
every logged channel against time: commanded CNC XYZ offsets, CNC B/C offsets,
head translation, and head pitch/yaw/roll. Each panel shows the per-bin mean
with a min/max envelope, plus the least-squares drift slope in the legend, so a
slow drift is visible even when it is buried in jitter.

  python scripts/figures/plot_stability.py --log results/cnc_stream_20260730_090000.csv

Out: results/figures/<log stem>_drift.{pdf,svg,png}
     results/<log stem>_drift_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

from figstyle import DOUBLE_COL_IN, PALETTE, apply_style, save_figure

from orbbec_head_tracking.cnc_stability_log import (
    HEAD_ROTATION_CHANNELS,
    HEAD_TRANSLATION_CHANNELS,
    OFFSET_ROTATION_CHANNELS,
    OFFSET_TRANSLATION_CHANNELS,
    ChannelSpec,
    DriftStats,
    channel_values,
    format_drift_table,
    load_stability_csv,
    summarize_drift,
    tracking_availability,
    valid_mask,
    write_drift_summary_csv,
)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO / "results" / "stability_live.csv"

_AXIS_COLORS = (
    PALETTE["vermillion"],
    PALETTE["green"],
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["purple"],
)


def _bin_stats(
    t: np.ndarray,
    values: np.ndarray,
    bins: int,
    *,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin mean/min/max on a fixed time grid; empty bins become NaN.

    A two-hour log at 10 Hz is ~72k samples per channel, which bloats vector
    output; binning keeps the envelope faithful at a fraction of the size.
    """
    centers = np.linspace(t_start, t_end, bins + 1)
    centers = 0.5 * (centers[:-1] + centers[1:])
    mean = np.full(bins, np.nan)
    lo = np.full(bins, np.nan)
    hi = np.full(bins, np.nan)

    finite = np.isfinite(t) & np.isfinite(values)
    if not np.any(finite) or t_end <= t_start:
        return centers, mean, lo, hi

    t_ok = t[finite]
    v_ok = values[finite]
    idx = np.clip(
        ((t_ok - t_start) / (t_end - t_start) * bins).astype(int), 0, bins - 1
    )
    counts = np.bincount(idx, minlength=bins).astype(float)
    sums = np.bincount(idx, weights=v_ok, minlength=bins)
    filled = counts > 0
    mean[filled] = sums[filled] / counts[filled]

    lo_acc = np.full(bins, np.inf)
    hi_acc = np.full(bins, -np.inf)
    np.minimum.at(lo_acc, idx, v_ok)
    np.maximum.at(hi_acc, idx, v_ok)
    lo[filled] = lo_acc[filled]
    hi[filled] = hi_acc[filled]
    return centers, mean, lo, hi


def _time_axis(t_s: np.ndarray) -> tuple[np.ndarray, str, float]:
    span = float(np.nanmax(t_s) - np.nanmin(t_s)) if t_s.size else 0.0
    if span >= 5400.0:
        return t_s / 3600.0, "time (h)", 3600.0
    if span >= 180.0:
        return t_s / 60.0, "time (min)", 60.0
    return t_s, "time (s)", 1.0


def _shade_dropouts(ax: plt.Axes, t_plot: np.ndarray, mask: np.ndarray) -> None:
    """Grey bands where the loop was not tracking / not baselined."""
    if mask.all() or t_plot.size == 0:
        return
    bad = ~mask
    edges = np.flatnonzero(np.diff(bad.astype(int)))
    starts = [0] if bad[0] else []
    ends: list[int] = []
    for e in edges:
        if bad[e + 1]:
            starts.append(e + 1)
        else:
            ends.append(e + 1)
    if len(ends) < len(starts):
        ends.append(len(bad))
    for s, e in zip(starts, ends):
        ax.axvspan(
            t_plot[s],
            t_plot[min(e, len(t_plot) - 1)],
            color=PALETTE["grey"],
            alpha=0.15,
            lw=0,
            zorder=0,
        )


def _panel(
    ax: plt.Axes,
    t_plot: np.ndarray,
    log: dict[str, np.ndarray],
    mask: np.ndarray,
    specs: tuple[ChannelSpec, ...],
    stats: dict[str, DriftStats],
    *,
    ylabel: str,
    bins: int,
    zero_at_start: bool,
) -> None:
    t_start = float(np.nanmin(t_plot)) if t_plot.size else 0.0
    t_end = float(np.nanmax(t_plot)) if t_plot.size else 1.0
    for spec, color in zip(specs, _AXIS_COLORS):
        if spec.key not in log:
            continue
        values = channel_values(log, spec.key, mask)
        stat = stats.get(spec.key)
        if zero_at_start and stat is not None and np.isfinite(stat.start_value):
            values = values - stat.start_value
        centers, mean, lo, hi = _bin_stats(
            t_plot, values, bins, t_start=t_start, t_end=t_end
        )
        label = spec.label.split()[-1]
        if stat is not None and np.isfinite(stat.slope_per_hour):
            label = f"{label} ({stat.slope_per_hour:+.3f} {spec.unit}/h)"
        ax.fill_between(centers, lo, hi, color=color, alpha=0.20, lw=0)
        ax.plot(centers, mean, color=color, lw=1.1, label=label)

    ax.axhline(0.0, color=PALETTE["grey"], lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(
        ncol=3,
        loc="upper right",
        frameon=False,
        fontsize=6.4,
        handlelength=1.3,
        columnspacing=0.8,
    )


def build(
    log: dict[str, np.ndarray],
    stats: list[DriftStats],
    *,
    bins: int = 1200,
    title: str = "",
) -> plt.Figure:
    apply_style()
    mask = valid_mask(log)
    t_plot, xlabel, _ = _time_axis(np.asarray(log["t_s"], dtype=float))
    by_key = {item.channel: item for item in stats}

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(DOUBLE_COL_IN, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    panels = (
        (OFFSET_TRANSLATION_CHANNELS, "CNC offset\nXYZ (mm)", False),
        (OFFSET_ROTATION_CHANNELS, "CNC offset\nB/C (deg)", False),
        (HEAD_TRANSLATION_CHANNELS, "head $\\Delta$ XYZ\n(mm, vs run start)", True),
        (HEAD_ROTATION_CHANNELS, "head $\\Delta$ P/Y/R\n(deg, vs run start)", True),
    )
    for ax, (specs, ylabel, zero_at_start) in zip(axes, panels):
        _panel(
            ax,
            t_plot,
            log,
            mask,
            specs,
            by_key,
            ylabel=ylabel,
            bins=bins,
            zero_at_start=zero_at_start,
        )
        _shade_dropouts(ax, t_plot, mask)

    axes[-1].set_xlabel(xlabel)
    if title:
        axes[0].set_title(title, fontsize=8.5)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log", type=str, default=str(_DEFAULT_LOG))
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Figure stem under results/figures (default: <log stem>_drift)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Drift summary CSV (default: <log>_drift_summary.csv)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=1200,
        help="Time bins per channel for the plotted mean/min-max envelope",
    )
    parser.add_argument(
        "--edge-window-sec",
        type=float,
        default=60.0,
        help="Averaging window at each end of the run for start/end values",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        parser.error(f"log not found: {log_path}")
    log = load_stability_csv(log_path)
    if len(log.get("t_s", [])) == 0:
        parser.error(f"log has no samples: {log_path}")

    stats = summarize_drift(log, edge_window_sec=args.edge_window_sec)
    t_s = np.asarray(log["t_s"], dtype=float)
    duration_h = float(np.nanmax(t_s) - np.nanmin(t_s)) / 3600.0
    availability = tracking_availability(log)

    print(f"{log_path.name}: {len(t_s)} samples, {duration_h:.2f} h, "
          f"tracking ok {availability * 100:.1f}%")
    print(format_drift_table(stats))

    summary_path = Path(args.summary) if args.summary else log_path.with_name(
        f"{log_path.stem}_drift_summary.csv"
    )
    write_drift_summary_csv(summary_path, stats)
    print(f"wrote {summary_path}")

    title = (
        f"Stability run: {duration_h:.2f} h, "
        f"tracking {availability * 100:.1f}% of samples"
    )
    fig = build(log, stats, bins=args.bins, title=title)
    name = args.name or f"{log_path.stem}_drift"
    for path in save_figure(fig, name):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
