from __future__ import annotations

import numpy as np

LANDMARK_MODEL_POINTS: dict[int, list[float]] = {
    1: [0.0, 0.0, 0.0],
    152: [0.0, -330.0, -65.0],
    33: [-225.0, 170.0, -135.0],
    263: [225.0, 170.0, -135.0],
    61: [-150.0, -150.0, -125.0],
    291: [150.0, -150.0, -125.0],
}
LANDMARK_INDICES: list[int] = list(LANDMARK_MODEL_POINTS)
FACE_3D_MODEL = np.array(
    [LANDMARK_MODEL_POINTS[index] for index in LANDMARK_INDICES],
    dtype=np.float32,
)
LANDMARK_WEIGHTS = np.array(
    [2.0, 1.0, 1.5, 1.5, 1.0, 1.0],
    dtype=np.float32,
)

EULER_EPSILON = 1e-6
AXIS_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],
        [120.0, 0.0, 0.0],
        [0.0, 120.0, 0.0],
        [0.0, 0.0, 120.0],
    ],
    dtype=np.float32,
)
