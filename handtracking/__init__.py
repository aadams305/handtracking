"""High-FPS 10-point hand tracking (MobileNetV4 + SimCC)."""

from handtracking.topology import (
    HAND_10_NAMES,
    MEDIAPIPE_TO_SLOT,
    mediapipe_indices_10,
)

__all__ = [
    "HAND_10_NAMES",
    "MEDIAPIPE_TO_SLOT",
    "mediapipe_indices_10",
]
