from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VisionStageTiming:
    frame_acquire_align_ms: float = 0.0
    landmark_ms: float = 0.0
    pose_estimate_ms: float = 0.0
    temporal_filter_ms: float = 0.0


@dataclass(frozen=True)
class LoopTimingSample:
    loop_index: int
    wall_time_sec: float
    loop_period_ms: float
    frame_acquire_align_ms: float
    landmark_ms: float
    pose_estimate_ms: float
    temporal_filter_ms: float
    encode_ms: float
    safety_rate_limit_ms: float
    udp_ms: float
    loop_total_ms: float
    tracking_ok: bool
    baseline_ready: bool
    face_detected: bool


LOOP_TIMING_FIELDNAMES: tuple[str, ...] = tuple(LoopTimingSample.__dataclass_fields__.keys())


def write_timing_csv(path: str | Path, samples: list[LoopTimingSample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOOP_TIMING_FIELDNAMES)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def summarize_timing(samples: list[LoopTimingSample]) -> dict[str, dict[str, float]]:
    if not samples:
        return {}

    numeric_fields = [
        "loop_period_ms",
        "frame_acquire_align_ms",
        "landmark_ms",
        "pose_estimate_ms",
        "temporal_filter_ms",
        "encode_ms",
        "safety_rate_limit_ms",
        "udp_ms",
        "loop_total_ms",
    ]
    summary: dict[str, dict[str, float]] = {}
    for field in numeric_fields:
        values = [float(getattr(sample, field)) for sample in samples]
        sorted_values = sorted(values)
        p95_index = max(0, min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1)))))
        summary[field] = {
            "count": float(len(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(sorted_values[0]),
            "p95": float(sorted_values[p95_index]),
            "max": float(sorted_values[-1]),
        }
    tracking_rate = sum(1 for sample in samples if sample.tracking_ok) / len(samples)
    summary["tracking_ok_rate"] = {"value": float(tracking_rate)}
    return summary


def format_summary_table(summary: dict[str, dict[str, float]]) -> str:
    lines = ["field,mean_ms,median_ms,p95_ms,max_ms"]
    for field, stats in summary.items():
        if field == "tracking_ok_rate":
            continue
        lines.append(
            f"{field},{stats['mean']:.3f},{stats['median']:.3f},{stats['p95']:.3f},{stats['max']:.3f}"
        )
    if "tracking_ok_rate" in summary:
        lines.append(f"tracking_ok_rate,{summary['tracking_ok_rate']['value']:.4f},,,")
    return "\n".join(lines)
