#!/usr/bin/env python3
"""Show MediaPipe face landmark detection on RGB."""

from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp

from capture import run_view_loop

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


def main() -> None:
    face_mesh = _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    def render(snapshot):
        canvas = snapshot.color_bgr.copy()
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                _mp_drawing.draw_landmarks(
                    canvas,
                    face_landmarks,
                    _mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=_mp_styles.get_default_face_mesh_tesselation_style(),
                )
                _mp_drawing.draw_landmarks(
                    canvas,
                    face_landmarks,
                    _mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=_mp_styles.get_default_face_mesh_contours_style(),
                )
            status = "Face landmarks detection from RGB - MediaPipe"
            color = (80, 220, 120)
        else:
            status = "No face detected"
            color = (80, 80, 255)
        cv2.putText(
            canvas,
            status,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        return canvas

    try:
        run_view_loop("Pipeline Demo: Face landmarks", render)
    finally:
        face_mesh.close()


if __name__ == "__main__":
    main()
