#!/usr/bin/env python3
"""Check Orbbec device visibility via pyorbbecsdk2 (no camera streaming)."""

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
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    for sub in (
        "",
        "extensions",
        "extensions/depthengine",
        "extensions/filters",
        "extensions/frameprocessor",
        "extensions/firmwareupdater",
    ):
        path = pkg / sub if sub else pkg
        if path.is_dir():
            add_dll_directory(str(path))


def main() -> int:
    _register_windows_dll_directories()
    try:
        from pyorbbecsdk import Context
    except ImportError as exc:
        print("pyorbbecsdk not installed. Run: pip install pyorbbecsdk2", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    count = Context().query_devices().get_count()
    print(f"Orbbec device count: {count}")
    if count == 0:
        print("Hint: close Orbbec Viewer, replug USB 3, install OrbbecSDK v2 if Viewer works but count is 0.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
