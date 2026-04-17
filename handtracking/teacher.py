"""MediaPipe Hands (Full) teacher for 10 keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from handtracking.topology import MEDIAPIPE_INDICES_10, NUM_HAND_JOINTS


@dataclass
class TeacherResult:
    ok: bool
    handedness: Optional[str]
    # NUM_HAND_JOINTS x (x_norm, y_norm, z_rel) in full image normalized coords [0,1] for x,y
    landmarks_norm: Optional[np.ndarray]


class MediaPipeTeacher:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import os
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} for MediaPipe Tasks API...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        self._mp = mp
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "MediaPipeTeacher":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def process_bgr(self, image_bgr: np.ndarray) -> TeacherResult:
        """Run on BGR uint8 image."""
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._detector.detect(mp_image)
        
        if not res.hand_landmarks:
            return TeacherResult(False, None, None)

        hand = res.hand_landmarks[0]
        handedness = None
        if res.handedness:
            handedness = res.handedness[0][0].category_name
            
        pts = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
        for i, idx in enumerate(MEDIAPIPE_INDICES_10):
            lm = hand[idx]
            pts[i, 0] = lm.x
            pts[i, 1] = lm.y
            pts[i, 2] = lm.z
        return TeacherResult(True, handedness, pts)


def extract_10_points_pixel(
    landmarks_norm: np.ndarray, width: int, height: int
) -> np.ndarray:
    """(NUM_HAND_JOINTS,3) normalized -> (NUM_HAND_JOINTS,2) pixel xy in source image."""
    out = np.zeros((NUM_HAND_JOINTS, 2), dtype=np.float32)
    out[:, 0] = landmarks_norm[:, 0] * width
    out[:, 1] = landmarks_norm[:, 1] * height
    return out
