#!/usr/bin/env python3
"""Show live native depth frames (before D2C alignment)."""

from __future__ import annotations

import cv2

from capture import run_view_loop


def main() -> None:
    def render(snapshot):
        canvas = snapshot.depth_native_vis.copy()
        h, w = snapshot.depth_native_mm.shape
        cv2.putText(
            canvas,
            f"Depth frames (native {w}x{h})",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        return canvas

    run_view_loop("Pipeline Demo: Depth frames", render)


if __name__ == "__main__":
    main()
