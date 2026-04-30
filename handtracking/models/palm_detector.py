"""BlazePalm-style lightweight palm detector for two-stage hand tracking.

Architecture: MobileNetV4ConvSmall backbone (shared design, width_mult=0.5 for speed)
→ SSD-style multi-scale detection head producing:
  - bounding box regression [cx, cy, w, h] (normalized 0..1 relative to input)
  - objectness confidence (sigmoid)

Input: 192×192 RGB (smaller than landmark model for speed).
Output: list of palm bounding boxes with confidence scores.

The detector finds palms, then the landmark SimCC model runs on the cropped region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handtracking.models.mobilenet_v4_conv_small import MobileNetV4ConvSmall

PALM_INPUT_SIZE = 192
NUM_ANCHORS_PER_CELL = 2


@dataclass
class PalmDetection:
    """A single palm detection in source image coordinates."""
    cx: float
    cy: float
    w: float
    h: float
    score: float
    angle: float = 0.0

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2

    def to_square(self, expand: float = 1.3) -> "PalmDetection":
        """Convert to a square bounding box with margin for the landmark model."""
        side = max(self.w, self.h) * expand
        return PalmDetection(
            cx=self.cx, cy=self.cy,
            w=side, h=side,
            score=self.score, angle=self.angle,
        )


def _generate_anchors(
    input_size: int = PALM_INPUT_SIZE,
    strides: tuple[int, ...] = (32, 16),
    anchors_per_cell: int = NUM_ANCHORS_PER_CELL,
) -> np.ndarray:
    """Pre-compute anchor centers (normalized 0..1) for each grid cell and stride."""
    anchors = []
    for stride in strides:
        grid = input_size // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(anchors_per_cell):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


class PalmDetHead(nn.Module):
    """Single-scale detection head: conv → box(4) + score(1) per anchor."""

    def __init__(self, in_channels: int, num_anchors: int = NUM_ANCHORS_PER_CELL) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.box_conv = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.cls_conv = nn.Conv2d(in_channels, num_anchors, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        boxes = self.box_conv(x).permute(0, 2, 3, 1).reshape(B, -1, 4)
        scores = self.cls_conv(x).permute(0, 2, 3, 1).reshape(B, -1)
        return boxes, scores


class BlazePalmDetector(nn.Module):
    """BlazePalm-style detector: backbone → two-scale detection heads.

    Returns raw box offsets and logits; use ``decode_detections`` for NMS post-processing.
    """

    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()
        self.backbone = MobileNetV4ConvSmall(width_mult=width_mult)
        ch = self.backbone.out_channels

        self.head_hi = PalmDetHead(ch, NUM_ANCHORS_PER_CELL)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU6(inplace=True),
        )
        self.head_lo = PalmDetHead(ch // 2, NUM_ANCHORS_PER_CELL)

        anchors = _generate_anchors(PALM_INPUT_SIZE, strides=(32, 16))
        self.register_buffer("anchors", torch.from_numpy(anchors))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (box_offsets [B, A, 4], score_logits [B, A])."""
        z = self.backbone(x)

        box_hi, cls_hi = self.head_hi(z)
        z_up = self.up(z)
        box_lo, cls_lo = self.head_lo(z_up)

        boxes = torch.cat([box_hi, box_lo], dim=1)
        scores = torch.cat([cls_hi, cls_lo], dim=1)
        return boxes, scores

    def decode(
        self,
        box_offsets: torch.Tensor,
        score_logits: torch.Tensor,
        score_thresh: float = 0.5,
        iou_thresh: float = 0.3,
    ) -> list[list[PalmDetection]]:
        """Decode raw outputs into NMS-filtered PalmDetection lists per batch item."""
        B = box_offsets.shape[0]
        anchors = self.anchors  # [A, 2]
        results: list[list[PalmDetection]] = []

        for b in range(B):
            scores = torch.sigmoid(score_logits[b])  # [A]
            offsets = box_offsets[b]  # [A, 4]

            cx = anchors[:, 0] + offsets[:, 0]
            cy = anchors[:, 1] + offsets[:, 1]
            w = offsets[:, 2].exp().clamp(max=2.0)
            h = offsets[:, 3].exp().clamp(max=2.0)

            mask = scores > score_thresh
            if not mask.any():
                results.append([])
                continue

            cx_f = cx[mask]
            cy_f = cy[mask]
            w_f = w[mask]
            h_f = h[mask]
            sc_f = scores[mask]

            x1 = cx_f - w_f / 2
            y1 = cy_f - h_f / 2
            x2 = cx_f + w_f / 2
            y2 = cy_f + h_f / 2
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            from torchvision.ops import nms
            keep = nms(bboxes, sc_f, iou_thresh)

            dets = []
            for k in keep:
                dets.append(PalmDetection(
                    cx=float(cx_f[k]), cy=float(cy_f[k]),
                    w=float(w_f[k]), h=float(h_f[k]),
                    score=float(sc_f[k]),
                ))
            results.append(dets)

        return results


def crop_palm_region(
    image_bgr: np.ndarray,
    det: PalmDetection,
    output_size: int = 256,
) -> tuple[np.ndarray, dict]:
    """Crop and resize the palm region to a square for the landmark model.

    Returns (cropped_image, transform_info) where transform_info contains
    the parameters needed to map landmarks back to the original image.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    sq = det.to_square(expand=1.4)

    x1 = int(sq.x1 * w)
    y1 = int(sq.y1 * h)
    x2 = int(sq.x2 * w)
    y2 = int(sq.y2 * h)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    crop = image_bgr[y1c:y2c, x1c:x2c]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    transform = {
        "x1": x1, "y1": y1,
        "crop_w": x2 - x1, "crop_h": y2 - y1,
        "output_size": output_size,
        "img_w": w, "img_h": h,
    }
    return resized, transform


def map_landmarks_to_source(
    landmarks_px: np.ndarray,
    transform: dict,
) -> np.ndarray:
    """Map landmarks from crop-space (output_size×output_size) back to source image."""
    out = landmarks_px.copy().astype(np.float32)
    s = transform["output_size"]
    out[:, 0] = out[:, 0] / s * transform["crop_w"] + transform["x1"]
    out[:, 1] = out[:, 1] / s * transform["crop_h"] + transform["y1"]
    return out


if __name__ == "__main__":
    model = BlazePalmDetector(width_mult=0.5)
    n = sum(p.numel() for p in model.parameters())
    print(f"Palm detector params: {n:,}")

    x = torch.randn(2, 3, PALM_INPUT_SIZE, PALM_INPUT_SIZE)
    boxes, scores = model(x)
    print(f"boxes={boxes.shape} scores={scores.shape}")

    dets = model.decode(boxes, scores, score_thresh=0.01)
    for i, d in enumerate(dets):
        print(f"  batch {i}: {len(d)} detections")
