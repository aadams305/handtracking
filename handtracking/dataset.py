"""SimCC training dataset from distilled JSONL manifest."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from handtracking.augmentations import cutout, motion_blur, rotate_180
from handtracking.dataset_manifest import iter_manifest
from handtracking.geometry import letterbox_image
from handtracking.models.hand_simcc import INPUT_SIZE
from handtracking.topology import NUM_HAND_JOINTS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_bgr_tensor(img_bgr_160: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ch = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (ch - mean) / std


def color_jitter(
    img: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    p: float = 0.5,
) -> np.ndarray:
    """Random color jitter augmentation in HSV space for domain robustness."""
    if random.random() >= p:
        return img
    img = img.copy().astype(np.float32)

    # Brightness
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = img * factor

    # Contrast
    if random.random() < 0.5:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean

    # Saturation (convert to HSV, scale S channel)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        factor = 1.0 + random.uniform(-saturation, saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def random_scale_crop(
    img: np.ndarray,
    kp: np.ndarray,
    scale_range: tuple[float, float] = (0.85, 1.15),
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Random scale within letterbox: zoom in/out around center, keeping hand visible."""
    if random.random() >= p:
        return img, kp
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    img_out = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

    kp_out = kp.copy()
    ones = np.ones((kp.shape[0], 1), dtype=np.float32)
    kp_h = np.hstack([kp_out, ones])  # [J, 3]
    kp_out = (M @ kp_h.T).T  # [J, 2]

    # Clamp keypoints to image bounds
    kp_out[:, 0] = np.clip(kp_out[:, 0], 0, w - 1)
    kp_out[:, 1] = np.clip(kp_out[:, 1], 0, h - 1)

    return img_out, kp_out.astype(np.float32)


class HandSimCCDataset(Dataset):
    """Dataset returning (image, keypoints, has_hand, handedness_label).

    handedness_label: 0.0 = Left, 1.0 = Right, 0.5 = unknown
    has_hand: 1.0 if hand present, 0.0 if negative sample
    """
    def __init__(
        self,
        manifest_path: Path,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.rows = list(iter_manifest(manifest_path))
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.rows[idx]
        img = cv2.imread(s.image_path)
        if img is None:
            raise FileNotFoundError(s.image_path)
        if s.letterbox.dst != INPUT_SIZE:
            raise ValueError(
                f"{s.image_path}: manifest letterbox.dst={s.letterbox.dst}; "
                f"expected {INPUT_SIZE} (re-run distill_freihand)."
            )
        lb_img, _ = letterbox_image(img, INPUT_SIZE)
        kp = np.array(s.keypoints_xy, dtype=np.float32)
        if kp.shape[0] != NUM_HAND_JOINTS:
            raise ValueError(
                f"{s.image_path}: manifest has {kp.shape[0]} keypoints; "
                f"expected {NUM_HAND_JOINTS}. Re-run handtracking.distill_freihand."
            )

        # Handedness: Left=0, Right=1, unknown=0.5
        if s.handedness == "Right":
            hand_label = 1.0
        elif s.handedness == "Left":
            hand_label = 0.0
        else:
            hand_label = 0.5  # unknown / don't supervise

        # Presence: all manifest samples have hands (has_hand defaults to True)
        has_hand = 1.0 if s.has_hand else 0.0

        if self.augment:
            # Color jitter for domain robustness
            lb_img = color_jitter(lb_img, p=0.5)

            if self._rng.random() < 0.5:
                lb_img = motion_blur(lb_img)
            lb_img, kp = rotate_180(lb_img, kp, p=0.25)

            # Random scale for handling different hand sizes
            lb_img, kp = random_scale_crop(lb_img, kp, p=0.4)

            # Horizontal flip — swaps handedness (left ↔ right)
            if self._rng.random() < 0.5:
                lb_img = cv2.flip(lb_img, 1)
                kp[:, 0] = INPUT_SIZE - 1 - kp[:, 0]
                # Swap handedness on flip
                if hand_label == 0.0:
                    hand_label = 1.0
                elif hand_label == 1.0:
                    hand_label = 0.0
                # 0.5 stays 0.5

            lb_img = cutout(lb_img, kp, p=0.4)

        inp = normalize_bgr_tensor(lb_img)
        target = torch.from_numpy(kp)
        has_hand_t = torch.tensor(has_hand, dtype=torch.float32)
        hand_label_t = torch.tensor(hand_label, dtype=torch.float32)
        return inp, target, has_hand_t, hand_label_t
