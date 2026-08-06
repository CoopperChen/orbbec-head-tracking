"""Long-run stability logging and drift analysis for the CNC compensation loop.

``StabilityCsvLogger`` streams one row per logged control tick straight to disk,
so a multi-hour session still leaves a usable CSV after Ctrl+C or a crash. The
column names overlap with ``scripts/figures/log_follow_mode.py`` so
``fig5_followmode.py`` can read the same file.

The analysis helpers turn such a log into per-channel drift statistics: a
least-squares slope per hour plus spread, which is what distinguishes a slowly
drifting offset from stationary jitter.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, TextIO

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .cnc_offset_encoder import CncUserOffset
    from .types import HeadPose

STABILITY_FIELDS: tuple[str, ...] = (
    "t_s",
    "wall_clock",
    "head_x_mm",
    "head_y_mm",
    "head_z_mm",
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "head_dx_mm",
    "head_dy_mm",
    "head_dz_mm",
    "req_x",
    "req_y",
    "req_z",
    "req_b",
    "req_c",
    "off_x",
    "off_y",
    "off_z",
    "off_b",
    "off_c",
    "confidence",
    "head_speed_mm_s",
    "loop_dt_ms",
    "step_mm",
    "step_deg",
    "tracking_ok",
    "baseline_ready",
    "link_ok",
    "safety_action",
    "safety_reason",
)

TEXT_FIELDS: frozenset[str] = frozenset({"wall_clock", "safety_action", "safety_reason"})


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    label: str
    unit: str


#: Channels reported by :func:`summarize_drift`, grouped as they are plotted.
OFFSET_TRANSLATION_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("off_x", "CNC offset X", "mm"),
    ChannelSpec("off_y", "CNC offset Y", "mm"),
    ChannelSpec("off_z", "CNC offset Z", "mm"),
)

OFFSET_ROTATION_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("off_b", "CNC offset B", "deg"),
    ChannelSpec("off_c", "CNC offset C", "deg"),
)

HEAD_TRANSLATION_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("head_x_mm", "head X", "mm"),
    ChannelSpec("head_y_mm", "head Y", "mm"),
    ChannelSpec("head_z_mm", "head Z", "mm"),
)

HEAD_ROTATION_CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("pitch_deg", "head pitch", "deg"),
    ChannelSpec("yaw_deg", "head yaw", "deg"),
    ChannelSpec("roll_deg", "head roll", "deg"),
)

DRIFT_CHANNELS: tuple[ChannelSpec, ...] = (
    OFFSET_TRANSLATION_CHANNELS
    + OFFSET_ROTATION_CHANNELS
    + HEAD_TRANSLATION_CHANNELS
    + HEAD_ROTATION_CHANNELS
)


class StabilityCsvLogger:
    """Rate-limited CSV writer for long CNC streaming sessions.

    ``rate_hz`` decimates the control loop (100 Hz would be ~720k rows over two
    hours); ``flush_sec`` bounds how much data an abrupt exit can lose.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rate_hz: float = 10.0,
        flush_sec: float = 5.0,
    ) -> None:
        self.path = Path(path)
        self.rate_hz = float(rate_hz)
        self._period_sec = 1.0 / self.rate_hz if self.rate_hz > 0.0 else 0.0
        self._flush_sec = max(0.0, float(flush_sec))
        self._handle: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._next_sample: float | None = None
        self._next_flush: float | None = None
        self.rows_written = 0

    def open(self) -> StabilityCsvLogger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=list(STABILITY_FIELDS),
            extrasaction="ignore",
            restval="",
        )
        self._writer.writeheader()
        self._handle.flush()
        return self

    def __enter__(self) -> StabilityCsvLogger:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        self._handle = None
        self._writer = None

    def due(self, now_sec: float) -> bool:
        """True when the next sample is owed at monotonic time ``now_sec``."""
        if self._period_sec <= 0.0:
            return True
        if self._next_sample is None:
            self._next_sample = now_sec
        if now_sec < self._next_sample:
            return False
        self._next_sample += self._period_sec
        if self._next_sample < now_sec:
            self._next_sample = now_sec + self._period_sec
        return True

    def record(
        self,
        *,
        now_sec: float,
        elapsed_sec: float,
        pose: HeadPose | None,
        head_delta_machine_mm: Sequence[float] | np.ndarray | None,
        required: CncUserOffset | None,
        sent: CncUserOffset,
        confidence: float,
        head_speed_mm_s: float,
        tracking_ok: bool,
        baseline_ready: bool,
        link_ok: bool,
        safety_action: str = "",
        safety_reason: str = "",
        loop_dt_ms: float | None = None,
        step_mm: float | None = None,
        step_deg: float | None = None,
    ) -> bool:
        """Write one row if the sample is due. Returns whether a row was written."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("StabilityCsvLogger.open() must be called before record()")
        if not self.due(now_sec):
            return False

        row: dict[str, object] = {
            "t_s": round(float(elapsed_sec), 4),
            "wall_clock": datetime.now().isoformat(timespec="milliseconds"),
            "off_x": round(float(sent.x), 4),
            "off_y": round(float(sent.y), 4),
            "off_z": round(float(sent.z), 4),
            "off_b": round(float(sent.b), 4),
            "off_c": round(float(sent.c), 4),
            "confidence": round(float(confidence), 4),
            "head_speed_mm_s": round(float(head_speed_mm_s), 3),
            "tracking_ok": int(bool(tracking_ok)),
            "baseline_ready": int(bool(baseline_ready)),
            "link_ok": int(bool(link_ok)),
            "safety_action": safety_action,
            "safety_reason": safety_reason,
        }
        if loop_dt_ms is not None:
            row["loop_dt_ms"] = round(float(loop_dt_ms), 2)
        if step_mm is not None:
            row["step_mm"] = round(float(step_mm), 4)
        if step_deg is not None:
            row["step_deg"] = round(float(step_deg), 4)
        if pose is not None:
            translation = np.asarray(pose.translation_vector_mm, dtype=np.float64).reshape(3)
            rvec = np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3)
            pitch, yaw, roll = pose.pitch_yaw_roll
            row.update(
                {
                    "head_x_mm": round(float(translation[0]), 3),
                    "head_y_mm": round(float(translation[1]), 3),
                    "head_z_mm": round(float(translation[2]), 3),
                    "pitch_deg": round(float(pitch), 3),
                    "yaw_deg": round(float(yaw), 3),
                    "roll_deg": round(float(roll), 3),
                    "rvec_x": round(float(rvec[0]), 6),
                    "rvec_y": round(float(rvec[1]), 6),
                    "rvec_z": round(float(rvec[2]), 6),
                }
            )
        if head_delta_machine_mm is not None:
            delta = np.asarray(head_delta_machine_mm, dtype=np.float64).reshape(3)
            row.update(
                {
                    "head_dx_mm": round(float(delta[0]), 3),
                    "head_dy_mm": round(float(delta[1]), 3),
                    "head_dz_mm": round(float(delta[2]), 3),
                }
            )
        if required is not None:
            row.update(
                {
                    "req_x": round(float(required.x), 4),
                    "req_y": round(float(required.y), 4),
                    "req_z": round(float(required.z), 4),
                    "req_b": round(float(required.b), 4),
                    "req_c": round(float(required.c), 4),
                }
            )

        self._writer.writerow(row)
        self.rows_written += 1
        if self._next_flush is None:
            self._next_flush = now_sec + self._flush_sec
        if now_sec >= self._next_flush:
            self._handle.flush()
            self._next_flush = now_sec + self._flush_sec
        return True


def load_stability_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a stability log into column arrays (floats, NaN for blanks)."""
    path = Path(path)
    columns: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            columns[name] = []
        for row in reader:
            for name in columns:
                columns[name].append((row.get(name) or "").strip())
    if not columns:
        raise ValueError(f"{path} has no columns")

    data: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name in TEXT_FIELDS:
            data[name] = np.asarray(values, dtype=object)
        else:
            data[name] = np.asarray(
                [float(v) if v else np.nan for v in values], dtype=float
            )
    return data


def valid_mask(log: dict[str, np.ndarray]) -> np.ndarray:
    """Samples where the loop was actually compensating (tracked + baselined)."""
    n = len(log["t_s"])
    mask = np.ones(n, dtype=bool)
    for key in ("tracking_ok", "baseline_ready"):
        if key in log:
            mask &= np.nan_to_num(log[key], nan=0.0) > 0.5
    return mask


def channel_values(
    log: dict[str, np.ndarray],
    key: str,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Channel as float array with invalid samples blanked to NaN."""
    values = np.asarray(log[key], dtype=float).copy()
    if mask is not None:
        values[~mask] = np.nan
    return values


def linear_drift_per_hour(t_s: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope in units per hour, ignoring NaN samples."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    if finite.sum() < 2:
        return float("nan")
    t_fit = t[finite]
    if np.ptp(t_fit) <= 0.0:
        return float("nan")
    slope_per_sec = float(np.polyfit(t_fit, v[finite], 1)[0])
    return slope_per_sec * 3600.0


@dataclass(frozen=True)
class DriftStats:
    channel: str
    label: str
    unit: str
    count: int
    mean: float
    std: float
    peak_to_peak: float
    max_abs: float
    start_value: float
    end_value: float
    slope_per_hour: float
    drift_over_run: float

    def as_row(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "label": self.label,
            "unit": self.unit,
            "count": self.count,
            "mean": round(self.mean, 5),
            "std": round(self.std, 5),
            "peak_to_peak": round(self.peak_to_peak, 5),
            "max_abs": round(self.max_abs, 5),
            "start_value": round(self.start_value, 5),
            "end_value": round(self.end_value, 5),
            "slope_per_hour": round(self.slope_per_hour, 5),
            "drift_over_run": round(self.drift_over_run, 5),
        }


DRIFT_SUMMARY_FIELDS: tuple[str, ...] = (
    "channel",
    "label",
    "unit",
    "count",
    "mean",
    "std",
    "peak_to_peak",
    "max_abs",
    "start_value",
    "end_value",
    "slope_per_hour",
    "drift_over_run",
)


def channel_drift(
    t_s: np.ndarray,
    values: np.ndarray,
    spec: ChannelSpec,
    *,
    edge_window_sec: float = 60.0,
) -> DriftStats:
    """Drift statistics for one channel.

    ``start_value`` / ``end_value`` average the first and last
    ``edge_window_sec`` of valid samples, so a single noisy sample at either end
    does not dominate the reported total drift.
    """
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        nan = float("nan")
        return DriftStats(spec.key, spec.label, spec.unit, 0, nan, nan, nan, nan, nan, nan, nan, nan)

    t_ok = t[finite]
    v_ok = v[finite]
    window = max(0.0, float(edge_window_sec))
    head = v_ok[t_ok <= t_ok[0] + window]
    tail = v_ok[t_ok >= t_ok[-1] - window]
    slope = linear_drift_per_hour(t_ok, v_ok)
    duration_hours = float(t_ok[-1] - t_ok[0]) / 3600.0
    return DriftStats(
        channel=spec.key,
        label=spec.label,
        unit=spec.unit,
        count=int(v_ok.size),
        mean=float(np.mean(v_ok)),
        std=float(np.std(v_ok)),
        peak_to_peak=float(np.ptp(v_ok)),
        max_abs=float(np.max(np.abs(v_ok))),
        start_value=float(np.mean(head)) if head.size else float(v_ok[0]),
        end_value=float(np.mean(tail)) if tail.size else float(v_ok[-1]),
        slope_per_hour=slope,
        drift_over_run=slope * duration_hours,
    )


def summarize_drift(
    log: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    channels: Sequence[ChannelSpec] = DRIFT_CHANNELS,
    edge_window_sec: float = 60.0,
) -> list[DriftStats]:
    if mask is None:
        mask = valid_mask(log)
    t_s = np.asarray(log["t_s"], dtype=float)
    stats: list[DriftStats] = []
    for spec in channels:
        if spec.key not in log:
            continue
        stats.append(
            channel_drift(
                t_s,
                channel_values(log, spec.key, mask),
                spec,
                edge_window_sec=edge_window_sec,
            )
        )
    return stats


def write_drift_summary_csv(path: str | Path, stats: Sequence[DriftStats]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DRIFT_SUMMARY_FIELDS))
        writer.writeheader()
        for item in stats:
            writer.writerow(item.as_row())
    return path


def format_drift_table(stats: Sequence[DriftStats]) -> str:
    header = f"{'channel':<14}{'unit':<6}{'mean':>10}{'std':>10}{'p2p':>10}{'drift/h':>10}{'total':>10}"
    lines = [header, "-" * len(header)]
    for item in stats:
        lines.append(
            f"{item.channel:<14}{item.unit:<6}"
            f"{item.mean:>10.3f}{item.std:>10.3f}{item.peak_to_peak:>10.3f}"
            f"{item.slope_per_hour:>10.3f}{item.drift_over_run:>10.3f}"
        )
    return "\n".join(lines)


def rolling_mean(t_s: np.ndarray, values: np.ndarray, window_sec: float) -> np.ndarray:
    """NaN-aware centred moving average over an approximately uniform time base."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.size == 0 or window_sec <= 0.0:
        return v
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    if not np.isfinite(dt) or dt <= 0.0:
        return v
    width = max(1, int(round(window_sec / dt)))
    if width <= 1:
        return v
    finite = np.isfinite(v)
    kernel = np.ones(width, dtype=float)
    sums = np.convolve(np.where(finite, v, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.full(v.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def tracking_availability(log: dict[str, np.ndarray]) -> float:
    """Fraction of logged samples with a valid tracked pose."""
    if "tracking_ok" not in log or len(log["tracking_ok"]) == 0:
        return float("nan")
    ok = np.nan_to_num(log["tracking_ok"], nan=0.0) > 0.5
    return float(np.count_nonzero(ok)) / float(ok.size)
