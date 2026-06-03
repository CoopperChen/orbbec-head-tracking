from __future__ import annotations

import numpy as np

from orbbec_head_tracking.smoothing import PoseSmoother


def test_pose_smoother_reset() -> None:
    smoother = PoseSmoother(0.2, 0.2, 1.0, 1.0)
    smoother.reset()
    assert smoother.translation_vector_mm is None
