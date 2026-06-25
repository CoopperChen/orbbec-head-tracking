"""Orbbec SDK v2 Python bindings.

Install package: ``pyorbbecsdk2`` (PyPI). Import module: ``pyorbbecsdk``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _register_windows_dll_directories() -> None:
    if sys.platform != "win32":
        return
    try:
        import pyorbbecsdk
    except ImportError:
        return
    pkg = Path(pyorbbecsdk.__file__).resolve().parent
    candidates = [
        pkg,
        pkg / "extensions",
        pkg / "extensions" / "depthengine",
        pkg / "extensions" / "filters",
        pkg / "extensions" / "frameprocessor",
        pkg / "extensions" / "firmwareupdater",
    ]
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    for path in candidates:
        if path.is_dir():
            add_dll_directory(str(path))


_register_windows_dll_directories()

from pyorbbecsdk import (  # noqa: E402
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

__all__ = [
    "AlignFilter",
    "Config",
    "Context",
    "OBFormat",
    "OBFrameAggregateOutputMode",
    "OBLogLevel",
    "OBSensorType",
    "OBStreamType",
    "Pipeline",
]
