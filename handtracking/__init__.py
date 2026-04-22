"""High-FPS 21-point hand tracking (MobileNetV4 + SimCC)."""

from handtracking.topology import (
    HAND_21_NAMES,
    MEDIAPIPE_TO_SLOT,
    mediapipe_indices_21,
)

__all__ = [
    "HAND_21_NAMES",
    "MEDIAPIPE_TO_SLOT",
    "mediapipe_indices_21",
]
