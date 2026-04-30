"""Train the BlazePalm detector on distilled manifest data.

Uses the same JSONL manifest as the landmark model — the bounding box ground truth
is derived from the 21-keypoint annotations (min/max of keypoints = palm bbox).

Usage:
    python -m handtracking.train_palm_detector --manifest data/distilled/manifest.jsonl --epochs 60
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from handtracking.dataset_manifest import iter_manifest
from handtracking.geometry import letterbox_image
from handtracking.models.palm_detector import (
    PALM_INPUT_SIZE,
    NUM_ANCHORS_PER_CELL,
    BlazePalmDetector,
    _generate_anchors,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def keypoints_to_bbox(kp: np.ndarray, img_size: int) -> tuple[float, float, float, float]:
    """Derive normalized (cx, cy, w, h) bbox from keypoints in pixel space [0, img_size)."""
    x_min, y_min = kp.min(axis=0)
    x_max, y_max = kp.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.15
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img_size, x_max + margin)
    y_max = min(img_size, y_max + margin)
    cx = (x_min + x_max) / 2.0 / img_size
    cy = (y_min + y_max) / 2.0 / img_size
    w = (x_max - x_min) / img_size
    h = (y_max - y_min) / img_size
    return cx, cy, w, h


class PalmDetDataset(Dataset):
    """Dataset yielding (image_tensor, bbox_target) for palm detection."""

    def __init__(self, manifest_path: Path, augment: bool = True) -> None:
        self.rows = list(iter_manifest(manifest_path))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.rows[idx]
        img = cv2.imread(s.image_path)
        if img is None:
            raise FileNotFoundError(s.image_path)

        lb_img, _ = letterbox_image(img, PALM_INPUT_SIZE)
        kp = np.array(s.keypoints_xy, dtype=np.float32)
        kp_scaled = kp * (PALM_INPUT_SIZE / s.letterbox.dst)

        if self.augment:
            if random.random() < 0.5:
                lb_img = cv2.flip(lb_img, 1)
                kp_scaled[:, 0] = PALM_INPUT_SIZE - 1 - kp_scaled[:, 0]

        cx, cy, w, h = keypoints_to_bbox(kp_scaled, PALM_INPUT_SIZE)

        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ch = torch.from_numpy(rgb).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        inp = (ch - mean) / std

        target = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        return inp, target


def assign_targets(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    pos_iou: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign ground truth boxes to anchors.

    gt_boxes: [B, 4] (cx, cy, w, h) normalized
    anchors: [A, 2] (cx, cy) normalized

    Returns:
        box_targets: [B, A, 4] offsets
        cls_targets: [B, A] (1.0 = positive, 0.0 = negative)
        pos_mask: [B, A] bool
    """
    B, A = gt_boxes.shape[0], anchors.shape[0]
    a_cx = anchors[:, 0].unsqueeze(0).expand(B, -1)
    a_cy = anchors[:, 1].unsqueeze(0).expand(B, -1)

    gt_cx = gt_boxes[:, 0:1].expand(-1, A)
    gt_cy = gt_boxes[:, 1:2].expand(-1, A)
    gt_w = gt_boxes[:, 2:3].expand(-1, A)
    gt_h = gt_boxes[:, 3:4].expand(-1, A)

    dist = ((a_cx - gt_cx) ** 2 + (a_cy - gt_cy) ** 2).sqrt()

    radius = (gt_w + gt_h) / 4.0
    pos_mask = dist < radius

    if not pos_mask.any():
        _, closest = dist.min(dim=1)
        for b in range(B):
            pos_mask[b, closest[b]] = True

    dx = gt_cx - a_cx
    dy = gt_cy - a_cy
    dw = gt_w.clamp(min=1e-4).log()
    dh = gt_h.clamp(min=1e-4).log()

    box_targets = torch.stack([dx, dy, dw, dh], dim=-1)
    cls_targets = pos_mask.float()

    return box_targets, cls_targets, pos_mask


def detection_loss(
    box_pred: torch.Tensor,
    cls_pred: torch.Tensor,
    box_targets: torch.Tensor,
    cls_targets: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Focal loss for classification + smooth L1 for positive box regression."""
    alpha, gamma = 0.25, 2.0
    p = torch.sigmoid(cls_pred)
    ce = F.binary_cross_entropy_with_logits(cls_pred, cls_targets, reduction="none")
    focal_weight = torch.where(
        cls_targets == 1,
        alpha * (1 - p) ** gamma,
        (1 - alpha) * p ** gamma,
    )
    cls_loss = (focal_weight * ce).mean()

    if pos_mask.any():
        box_loss = F.smooth_l1_loss(
            box_pred[pos_mask], box_targets[pos_mask], beta=0.1,
        )
    else:
        box_loss = torch.tensor(0.0, device=box_pred.device)

    return cls_loss + 2.0 * box_loss


def main() -> None:
    ap = argparse.ArgumentParser(description="Train BlazePalm detector")
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/palm_det.pt"))
    ap.add_argument("--width-mult", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ds = PalmDetDataset(args.manifest, augment=True)
    print(f"Palm dataset: {len(ds)} samples", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = BlazePalmDetector(width_mult=args.width_mult).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Palm detector params: {n_params:,}", flush=True)

    anchors = _generate_anchors(PALM_INPUT_SIZE)
    anchors_t = torch.from_numpy(anchors).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for imgs, gt in loader:
            imgs = imgs.to(device)
            gt = gt.to(device)

            box_pred, cls_pred = model(imgs)
            box_tgt, cls_tgt, pos_mask = assign_targets(gt, anchors_t)
            loss = detection_loss(box_pred, cls_pred, box_tgt, cls_tgt, pos_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        sched.step()
        avg_loss = total_loss / max(1, n)
        marker = " *best*" if avg_loss < best_loss else ""
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.6f}{marker}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "width_mult": args.width_mult,
            }, args.out)

    print(f"Saved best palm detector to {args.out}")


if __name__ == "__main__":
    main()
