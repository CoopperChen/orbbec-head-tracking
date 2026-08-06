from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from orbbec_head_tracking.stream_cnc import (
    DEFAULT_LOG_STEM,
    _parse_args,
    _resolve_log_path,
)

STAMPED = re.compile(r"_\d{8}_\d{6}\.csv$")


def _args(*argv: str):
    sys.argv = ["orbbec-head-stream-cnc", *argv]
    return _parse_args()


def test_no_log_flag_disables_logging() -> None:
    assert _resolve_log_path(_args()) is None


def test_bare_log_uses_default_timestamped_path() -> None:
    path = _resolve_log_path(_args("--log"))
    assert path is not None
    assert path.parent == Path("results")
    assert path.stem.startswith(f"{DEFAULT_LOG_STEM}_")
    assert STAMPED.search(path.name)


def test_log_with_csv_path_is_timestamped() -> None:
    path = _resolve_log_path(_args("--log", "out/run.csv"))
    assert path is not None
    assert path.parent == Path("out")
    assert path.stem.startswith("run_")
    assert STAMPED.search(path.name)


def test_log_with_directory_uses_default_stem(tmp_path: Path) -> None:
    path = _resolve_log_path(_args("--log", str(tmp_path)))
    assert path is not None
    assert path.parent == tmp_path
    assert path.stem.startswith(f"{DEFAULT_LOG_STEM}_")


def test_log_csv_path_is_verbatim_unless_timestamped() -> None:
    assert _resolve_log_path(_args("--log-csv", "results/stability.csv")) == Path(
        "results/stability.csv"
    )
    stamped = _resolve_log_path(_args("--log-csv", "results/stability.csv", "--log-timestamped"))
    assert stamped is not None
    assert STAMPED.search(stamped.name)


def test_log_and_log_csv_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _args("--log", "--log-csv", "results/stability.csv")
