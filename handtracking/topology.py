"""
10-point topology: subset of MediaPipe Hands 21 landmarks.

Slot order (model output index 0..9):
  Wrist, MCPs (index..pinky), tips (thumb..pinky).
  Middle PIP (former 11th point / MediaPipe index 10) is omitted.
"""

from __future__ import annotations

from typing import Final, List, Tuple

NUM_HAND_JOINTS = 10

# Output slot index -> MediaPipe landmark index
MEDIAPIPE_INDICES_10: Final[Tuple[int, ...]] = (
    0,  # wrist
    5,
    9,
    13,
    17,  # MCPs index, middle, ring, pinky
    4,
    8,
    12,
    16,
    20,  # tips thumb..pinky
)

HAND_10_NAMES: Final[Tuple[str, ...]] = (
    "wrist",
    "mcp_index",
    "mcp_middle",
    "mcp_ring",
    "mcp_pinky",
    "tip_thumb",
    "tip_index",
    "tip_middle",
    "tip_ring",
    "tip_pinky",
)

MEDIAPIPE_TO_SLOT: Final[dict[int, int]] = {
    mp: slot for slot, mp in enumerate(MEDIAPIPE_INDICES_10)
}


def mediapipe_indices_10() -> List[int]:
    return list(MEDIAPIPE_INDICES_10)
