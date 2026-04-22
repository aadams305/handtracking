"""
Full 21-point MediaPipe hand topology.
"""

from __future__ import annotations

from typing import Final, List, Tuple

NUM_HAND_JOINTS = 21

# Output slot index -> MediaPipe landmark index (1-to-1)
MEDIAPIPE_INDICES_21: Final[Tuple[int, ...]] = tuple(range(21))

HAND_21_NAMES: Final[Tuple[str, ...]] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

MEDIAPIPE_TO_SLOT: Final[dict[int, int]] = {
    mp: slot for slot, mp in enumerate(MEDIAPIPE_INDICES_21)
}

def mediapipe_indices_21() -> List[int]:
    return list(MEDIAPIPE_INDICES_21)
