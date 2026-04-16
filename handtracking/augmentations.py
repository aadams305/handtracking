"""Training augmentations: motion blur, 180° rotation, cutout."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np


def motion_blur(
    image_bgr: np.ndarray, max_kernel: int = 15, p: float = 0.5
) -> np.ndarray:
    """Simulate high-speed movement with directional motion blur."""
    import cv2

    if random.random() >= p:
        return image_bgr
    k = random.choice(range(3, max_kernel + 1, 2))
    kernel = np.zeros((k, k), dtype=np.float32)
    angle = random.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    kernel[int(k / 2), :] = c
    kernel[:, int(k / 2)] += s
    kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
    return cv2.filter2D(image_bgr, -1, kernel)


def rotate_180(image_bgr: np.ndarray, keypoints_xy: np.ndarray, p: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Random 180° rotation (image + keypoints in pixel space, shape H,W)."""
    import cv2

    if random.random() >= p:
        return image_bgr, keypoints_xy
    h, w = image_bgr.shape[:2]
    out = cv2.rotate(image_bgr, cv2.ROTATE_180)
    kp = keypoints_xy.copy()
    kp[:, 0] = w - 1 - kp[:, 0]
    kp[:, 1] = h - 1 - kp[:, 1]
    return out, kp


def cutout(
    image_bgr: np.ndarray,
    keypoints_xy: np.ndarray,
    num_patches: int = 2,
    max_frac: float = 0.15,
    p: float = 0.4,
) -> np.ndarray:
    """Random rectangular erasing (finger-on-finger occlusion)."""
    import cv2

    if random.random() >= p:
        return image_bgr
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for _ in range(num_patches):
        bh = int(random.uniform(h * 0.05, h * max_frac))
        bw = int(random.uniform(w * 0.05, w * max_frac))
        cx = random.randint(0, max(0, w - bw))
        cy = random.randint(0, max(0, h - bh))
        out[cy : cy + bh, cx : cx + bw] = random.randint(0, 255)
    return out
