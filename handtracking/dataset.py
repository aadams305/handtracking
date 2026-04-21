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
from handtracking.topology import NUM_HAND_JOINTS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_bgr_tensor(img_bgr_160: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ch = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (ch - mean) / std


class HandSimCCDataset(Dataset):
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.rows[idx]
        img = cv2.imread(s.image_path)
        if img is None:
            raise FileNotFoundError(s.image_path)
        lb_img, _ = letterbox_image(img, 160)
        kp = np.array(s.keypoints_xy, dtype=np.float32)
        if kp.shape[0] != NUM_HAND_JOINTS:
            raise ValueError(
                f"{s.image_path}: manifest has {kp.shape[0]} keypoints; "
                f"expected {NUM_HAND_JOINTS}. Re-run handtracking.distill_freihand."
            )
        if self.augment:
            if self._rng.random() < 0.5:
                lb_img = motion_blur(lb_img)
            lb_img, kp = rotate_180(lb_img, kp, p=0.25)
            
            # Massive Domain Gap Fix: Aggressive Scale & Translation
            if self._rng.random() < 0.8:
                # Random scale between 0.3x and 1.0x (simulating far away webcams)
                scale = self._rng.uniform(0.3, 1.0)
                # Random translation to corners
                tx = self._rng.uniform(-160 * (1 - scale), 160 * (1 - scale))
                ty = self._rng.uniform(-160 * (1 - scale), 160 * (1 - scale))
                
                M = np.float32([[scale, 0, tx], [0, scale, ty]])
                lb_img = cv2.warpAffine(lb_img, M, (160, 160), borderValue=(114, 114, 114))
                # Update keypoints
                kp = kp * scale + np.array([tx, ty], dtype=np.float32)
                
            lb_img = cutout(lb_img, kp, p=0.4)
        inp = normalize_bgr_tensor(lb_img)
        target = torch.from_numpy(kp)
        return inp, target
