"""Figure 4 - 6-DoF head-tracking method (3 panels).

(a) MediaPipe FaceMesh landmarks on the RGB image, highlighting the six model
    landmarks used for pose. (b) Those landmarks back-projected to 3D via the
    aligned depth (the real FACE_3D_MODEL anchor points). (c) The resulting
    6-DoF pose drawn as a coordinate triad on the face.

Uses the real constants from orbbec_head_tracking (FACE_3D_MODEL, AXIS_3D_MODEL,
LANDMARK_INDICES). Panel (a) can overlay a real capture via --frames <npz>
containing 'color_bgr' and 'landmarks_2d'.

Run:  python scripts/figures/fig4_tracking.py
Out:  results/figures/fig4_tracking.{pdf,svg,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from figstyle import DOUBLE_COL_IN, PALETTE, ROLE, apply_style, save_figure

from orbbec_head_tracking.constants import AXIS_3D_MODEL, FACE_3D_MODEL, LANDMARK_INDICES

# 2D layout of the six model landmarks for the schematic face (image-like coords).
_FACE_2D = {
    1: (0.0, 0.0),      # nose tip
    152: (0.0, -2.3),   # chin
    33: (-1.25, 1.15),  # left eye outer
    263: (1.25, 1.15),  # right eye outer
    61: (-0.95, -1.05), # left mouth corner
    291: (0.95, -1.05), # right mouth corner
}
_LM_LABEL = {1: "1", 152: "152", 33: "33", 263: "263", 61: "61", 291: "291"}


def _panel_a(ax: plt.Axes, frames: str | None) -> None:
    ax.set_title("(a) FaceMesh landmarks (RGB)", fontsize=8.5)
    if frames:
        data = np.load(frames)
        if "color_bgr" in data:
            img = data["color_bgr"][..., ::-1]  # BGR -> RGB
            ax.imshow(img, aspect="equal")
        if "landmarks_2d" in data:
            pts = np.asarray(data["landmarks_2d"], dtype=float)
            ax.scatter(pts[:, 0], pts[:, 1], s=3, color=PALETTE["green"], alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.4, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    # Face + light 468-point mesh suggestion.
    ax.add_patch(Ellipse((0, -0.3), 4.4, 6.0, facecolor=PALETTE["light"],
                         edgecolor=PALETTE["grey"], lw=1.0, zorder=1))
    rng = np.random.default_rng(7)
    n = 320
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    mx = r * np.cos(theta) * 2.05
    my = r * np.sin(theta) * 2.85 - 0.3
    ax.scatter(mx, my, s=1.2, color=PALETTE["sky"], alpha=0.5, zorder=2)
    # Six model landmarks.
    for idx in LANDMARK_INDICES:
        x, y = _FACE_2D[idx]
        ax.scatter([x], [y], s=34, color=PALETTE["vermillion"], edgecolor="black",
                   lw=0.5, zorder=4)
        ax.annotate(_LM_LABEL[idx], (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6.5, color=PALETTE["vermillion"])
    ax.text(0, -3.25, "6 anchor landmarks (of 468)", ha="center", fontsize=6.6,
            color=PALETTE["vermillion"], style="italic")


def _panel_b(ax: plt.Axes) -> None:
    ax.set_title("(b) Depth back-projection (3D)", fontsize=8.5, pad=0.0)
    pts = np.asarray(FACE_3D_MODEL, dtype=float)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    ax.scatter(xs, ys, zs, s=36, color=PALETTE["vermillion"], edgecolor="black",
               depthshade=False, lw=0.4)
    idx = {v: i for i, v in enumerate(LANDMARK_INDICES)}
    links = [(33, 263), (33, 61), (263, 291), (61, 291), (1, 152), (33, 1), (263, 1)]
    for a, b in links:
        ia, ib = idx[a], idx[b]
        ax.plot([xs[ia], xs[ib]], [ys[ia], ys[ib]], [zs[ia], zs[ib]],
                color=PALETTE["blue"], lw=1.0, alpha=0.8)
    for i, v in enumerate(LANDMARK_INDICES):
        ax.text(xs[i], ys[i], zs[i], f" {v}", fontsize=6.0, color=PALETTE["black"])
    ax.set_xlabel("X (mm)", fontsize=7, labelpad=-6)
    ax.set_ylabel("Y (mm)", fontsize=7, labelpad=-6)
    ax.set_zlabel("Z (mm)", fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=18, azim=-72)
    ax.text2D(0.5, -0.02, "rigid fit to aligned-depth points", transform=ax.transAxes,
              ha="center", fontsize=6.6, color=PALETTE["blue"], style="italic")


def _panel_c(ax: plt.Axes) -> None:
    ax.set_title("(c) 6-DoF pose", fontsize=8.5)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.4, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Ellipse((0, -0.3), 4.4, 6.0, facecolor=PALETTE["light"],
                         edgecolor=PALETTE["grey"], lw=1.0, zorder=1))
    for idx in LANDMARK_INDICES:
        x, y = _FACE_2D[idx]
        ax.scatter([x], [y], s=14, color=PALETTE["grey"], zorder=2)

    # Pose triad projected at the nose (X red, Y green, Z blue), matching viz.py.
    origin = np.array([0.0, 0.0])
    axis_defs = [
        (np.array([1.6, -0.35]), PALETTE["vermillion"], "X"),
        (np.array([0.15, -1.7]), PALETTE["green"], "Y"),
        (np.array([-0.7, 0.6]), PALETTE["blue"], "Z"),
    ]
    for vec, color, label in axis_defs:
        ax.annotate("", xy=origin + vec, xytext=origin,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        tip = origin + vec * 1.12
        ax.text(tip[0], tip[1], label, color=color, fontsize=8, fontweight="bold",
                ha="center", va="center")
    ax.scatter([0], [0], s=20, color="black", zorder=5)

    readout = ("X = -12.4 mm   Y = 8.1 mm   Z = 612 mm\n"
               "pitch = -6.2 deg   yaw = 11.4 deg   roll = 1.8 deg")
    ax.text(0, -3.15, readout, ha="center", fontsize=6.4, color=PALETTE["black"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PALETTE["grey"], lw=0.6))


def build(frames: str | None = None) -> plt.Figure:
    apply_style()
    fig = plt.figure(figsize=(DOUBLE_COL_IN, 2.5))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2, projection="3d")
    ax_c = fig.add_subplot(1, 3, 3)
    _panel_a(ax_a, frames)
    _panel_b(ax_b)
    _panel_c(ax_c)
    fig.tight_layout(pad=0.5, w_pad=1.0)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=str, default=None,
                        help="Optional .npz with 'color_bgr' and 'landmarks_2d' for panel (a)")
    args = parser.parse_args()
    fig = build(args.frames)
    for p in save_figure(fig, "fig4_tracking"):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
