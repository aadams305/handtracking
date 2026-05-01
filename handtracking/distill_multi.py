"""Distill multiple hand datasets (FreiHAND, RHD, OneHand10K) through MediaPipe → unified JSONL manifest.

Supports three dataset formats:
  - FreiHAND:    <root>/training/rgb/*.jpg (or *.png)
  - RHD:         <root>/training/color/*.png  (Rendered Handpose Dataset)
  - OneHand10K:  <root>/images/*.jpg + <root>/annotations/*.json (COCO-style)

Usage:
    python -m handtracking.distill_multi \
        --freihand /path/to/FreiHAND \
        --rhd /path/to/RHD \
        --onehand10k /path/to/OneHand10K \
        --out data/distilled/manifest_multi.jsonl

You can omit any dataset you don't have; at least one is required.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from handtracking.dataset_manifest import DistilledSample, LetterboxMeta, write_manifest
from handtracking.geometry import letterbox_image
from handtracking.teacher import MediaPipeTeacher, extract_21_points_pixel


DST_SIZE = 256


def _distill_image_list(
    teacher: MediaPipeTeacher,
    image_paths: list[Path],
    dataset_tag: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[DistilledSample]:
    """Run MediaPipe on a list of images, returning distilled samples."""
    rng = random.Random(seed)
    if max_samples and len(image_paths) > max_samples:
        image_paths = rng.sample(image_paths, max_samples)

    samples: list[DistilledSample] = []
    skipped = 0
    for p in tqdm(image_paths, desc=f"distill-{dataset_tag}", file=sys.stderr, mininterval=1.0):
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        tr = teacher.process_bgr(img)
        if not tr.ok or tr.landmarks_norm is None:
            skipped += 1
            continue

        kp_src = extract_21_points_pixel(tr.landmarks_norm[:, :2], w, h)
        lb_img, lb = letterbox_image(img, DST_SIZE)
        kp_dst = np.zeros((21, 2), dtype=np.float32)
        for i in range(21):
            kp_dst[i, 0], kp_dst[i, 1] = lb.map_xy_src_to_dst(
                float(kp_src[i, 0]), float(kp_src[i, 1])
            )
        meta = LetterboxMeta(
            scale=lb.scale, pad_x=lb.pad_x, pad_y=lb.pad_y,
            src_w=lb.src_w, src_h=lb.src_h, dst=DST_SIZE,
        )
        samples.append(DistilledSample(
            image_path=str(p.resolve()),
            letterbox=meta,
            keypoints_xy=kp_dst.tolist(),
            handedness=tr.handedness,
            has_hand=True,
        ))

    print(f"  [{dataset_tag}] Distilled {len(samples)}/{len(image_paths)} (skipped {skipped})", flush=True)
    return samples


def find_freihand_images(root: Path) -> list[Path]:
    """Find FreiHAND images under root."""
    for candidate in (root / "training" / "rgb", root / "rgb", root):
        if candidate.is_dir():
            imgs = sorted(candidate.glob("*.jpg")) + sorted(candidate.glob("*.png"))
            if imgs:
                return imgs
    return []


def find_rhd_images(root: Path) -> list[Path]:
    """Find RHD (Rendered Handpose Dataset) images.

    Handles both flat and nested zip layouts:
      data/rhd/training/color/*.png            (flat)
      data/rhd/RHD_published_v2/training/color/*.png  (nested from zip)
    """
    paths: list[Path] = []
    candidates = [root]
    for child in root.iterdir():
        if child.is_dir() and "rhd" in child.name.lower():
            candidates.append(child)
    for base in candidates:
        for split in ("training", "evaluation"):
            color_dir = base / split / "color"
            if color_dir.is_dir():
                paths.extend(sorted(color_dir.glob("*.png")))
    if not paths:
        for d in (root, root / "color"):
            if d.is_dir():
                paths.extend(sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")))
    return paths


def find_onehand10k_images(root: Path) -> list[Path]:
    """Find OneHand10K images.

    OneHand10K structure: Train/source/*.jpg, Test/source/*.jpg
    or: images/*.jpg
    """
    paths: list[Path] = []
    for sub in ("Train/source", "Test/source", "images"):
        d = root / sub
        if d.is_dir():
            paths.extend(sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")))
    if not paths:
        paths = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-dataset distillation → unified manifest")
    ap.add_argument("--freihand", type=Path, default=None, help="FreiHAND dataset root")
    ap.add_argument("--rhd", type=Path, default=None, help="RHD dataset root")
    ap.add_argument("--onehand10k", type=Path, default=None, help="OneHand10K dataset root")
    ap.add_argument("--out", type=Path, default=Path("data/distilled/manifest_multi.jsonl"))
    ap.add_argument("--max-per-dataset", type=int, default=None,
                     help="Max samples per dataset (for quick testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    datasets: list[tuple[str, list[Path]]] = []

    if args.freihand is not None:
        imgs = find_freihand_images(args.freihand)
        if imgs:
            datasets.append(("freihand", imgs))
            print(f"FreiHAND: {len(imgs)} images found", flush=True)
        else:
            print(f"WARNING: No FreiHAND images found under {args.freihand}", file=sys.stderr)

    if args.rhd is not None:
        imgs = find_rhd_images(args.rhd)
        if imgs:
            datasets.append(("rhd", imgs))
            print(f"RHD: {len(imgs)} images found", flush=True)
        else:
            print(f"WARNING: No RHD images found under {args.rhd}", file=sys.stderr)

    if args.onehand10k is not None:
        imgs = find_onehand10k_images(args.onehand10k)
        if imgs:
            datasets.append(("onehand10k", imgs))
            print(f"OneHand10K: {len(imgs)} images found", flush=True)
        else:
            print(f"WARNING: No OneHand10K images found under {args.onehand10k}", file=sys.stderr)

    if not datasets:
        print("ERROR: No datasets specified. Provide at least one of --freihand, --rhd, --onehand10k",
              file=sys.stderr)
        sys.exit(1)

    total_imgs = sum(len(imgs) for _, imgs in datasets)
    print(f"\nTotal images across {len(datasets)} dataset(s): {total_imgs}", flush=True)
    print("Loading MediaPipe (this may take several minutes on first run)...", flush=True)

    all_samples: list[DistilledSample] = []
    with MediaPipeTeacher() as teacher:
        for tag, imgs in datasets:
            samples = _distill_image_list(
                teacher, imgs, tag,
                max_samples=args.max_per_dataset,
                seed=args.seed,
            )
            all_samples.extend(samples)

    random.Random(args.seed).shuffle(all_samples)
    write_manifest(all_samples, args.out)
    print(f"\nWrote {len(all_samples)} total samples to {args.out}", flush=True)

    per_dataset: dict[str, int] = {}
    for s in all_samples:
        path = s.image_path.lower()
        for tag, _ in datasets:
            if tag in path:
                per_dataset[tag] = per_dataset.get(tag, 0) + 1
                break
    for tag, count in per_dataset.items():
        print(f"  {tag}: {count} samples", flush=True)


if __name__ == "__main__":
    main()
