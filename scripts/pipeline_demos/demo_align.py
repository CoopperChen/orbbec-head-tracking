#!/usr/bin/env python3
"""Show depth-to-color registration: RGB vs aligned depth vs overlay."""

from __future__ import annotations

import cv2
import numpy as np

from capture import blend_rgb_depth, run_view_loop


def main() -> None:
    def render(snapshot):
        rgb = snapshot.color_bgr
        aligned = snapshot.depth_aligned_vis
        overlay = blend_rgb_depth(rgb, snapshot.depth_aligned_mm)
        h, w = snapshot.depth_aligned_mm.shape
        panel = np.hstack([rgb, aligned, overlay])
        cv2.putText(
            panel,
            "RGB",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Align depth with RGB ({w}x{h})",
            (w + 16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Overlay",
            (2 * w + 16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        return panel

    run_view_loop("Pipeline Demo: Align depth with RGB", render)


if __name__ == "__main__":
    main()
