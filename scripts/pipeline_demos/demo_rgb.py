#!/usr/bin/env python3
"""Show live RGB frames from the Orbbec Gemini 2L."""

from __future__ import annotations

import cv2

from capture import run_view_loop


def main() -> None:
    def render(snapshot):
        canvas = snapshot.color_bgr.copy()
        cv2.putText(
            canvas,
            "RGB frames",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        return canvas

    run_view_loop("Pipeline Demo: RGB frames", render)


if __name__ == "__main__":
    main()
