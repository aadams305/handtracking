"""Draw hand skeleton on BGR image (10-slot legacy layout or full 21 MediaPipe joints)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Edges for visualization (slot indices 0..9); middle finger MCP->tip direct (no PIP)
EDGES_10: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),  # wrist to thumb tip
    (1, 6),
    (2, 7),
    (3, 8),
    (4, 9),
]

# MediaPipe-style 21-landmark connectivity (indices in NUM_HAND_JOINTS order).
EDGES_21: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_hand_10(
    image_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    radius: int = 3,
    color_pt: Tuple[int, int, int] = (0, 255, 0),
    color_edge: Tuple[int, int, int] = (255, 128, 0),
) -> np.ndarray:
    import cv2

    out = image_bgr.copy()
    n = keypoints_xy.shape[0]
    for i, j in EDGES_10:
        if i < n and j < n:
            p1 = (int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1]))
            p2 = (int(keypoints_xy[j, 0]), int(keypoints_xy[j, 1]))
            cv2.line(out, p1, p2, color_edge, 1, cv2.LINE_AA)
    for k in range(n):
        c = (int(keypoints_xy[k, 0]), int(keypoints_xy[k, 1]))
        cv2.circle(out, c, radius, color_pt, -1, cv2.LINE_AA)
    return out


def draw_hand_21(
    image_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    radius: int = 3,
    color_pt: Tuple[int, int, int] = (0, 255, 0),
    color_edge: Tuple[int, int, int] = (255, 128, 0),
) -> np.ndarray:
    import cv2

    out = image_bgr.copy()
    n = keypoints_xy.shape[0]
    for i, j in EDGES_21:
        if i < n and j < n:
            p1 = (int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1]))
            p2 = (int(keypoints_xy[j, 0]), int(keypoints_xy[j, 1]))
            cv2.line(out, p1, p2, color_edge, 1, cv2.LINE_AA)
    for k in range(n):
        c = (int(keypoints_xy[k, 0]), int(keypoints_xy[k, 1]))
        cv2.circle(out, c, radius, color_pt, -1, cv2.LINE_AA)
    return out
