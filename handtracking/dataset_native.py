"""Native-label dataset for FreiHAND + RHD (no MediaPipe distillation).

Loads annotations directly from dataset files:
  - FreiHAND: training_xyz.json + training_K.json -> 3D-to-2D projection
  - RHD: anno_{training,evaluation}.mat -> uv_vis (already 2D pixel coords)

Images are letterboxed to 256x256 and keypoints are mapped to letterbox space.
Normalization uses RTMPose pixel-space mean/std (not ImageNet /255 convention).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from handtracking.augmentations import cutout, motion_blur, random_rotation, rotate_180
from handtracking.dataset import color_jitter, random_scale_crop
from handtracking.geometry import letterbox_image
from handtracking.models.rtmpose_hand import INPUT_SIZE, MEAN, STD
from handtracking.topology import NUM_HAND_JOINTS


def normalize_rtmpose(img_bgr: np.ndarray) -> torch.Tensor:
    """Pixel-space normalization matching RTMPose convention.

    Input: BGR uint8 [H, W, 3]. Output: float32 [3, H, W] RGB normalised.
    mean/std are in pixel [0,255] space, applied to RGB channels.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    ch = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    return (ch - mean) / std


def _project_3d_to_2d(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 21x3 world coords to 21x2 pixel coords using 3x3 intrinsic K."""
    uv_h = (K @ xyz.T).T  # [21, 3]
    return uv_h[:, :2] / uv_h[:, 2:3]  # [21, 2]


def _augment_sample(
    img: np.ndarray,
    kp: np.ndarray,
    rng: random.Random,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply training augmentations (shared between FreiHAND and RHD)."""
    img = color_jitter(img, p=0.5)
    if rng.random() < 0.5:
        img = motion_blur(img)
    img, kp = rotate_180(img, kp, p=0.25)
    img, kp = random_rotation(img, kp, max_angle=30.0, p=0.5)
    img, kp = random_scale_crop(img, kp, p=0.4)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        kp[:, 0] = input_size - 1 - kp[:, 0]
    img = cutout(img, kp, p=0.4)
    return img, kp


class FreiHANDDataset(Dataset):
    """FreiHAND native labels.

    Args:
        root: Path containing ``training/rgb/``, ``training_xyz.json``, ``training_K.json``.
        augment: Enable training augmentations.
    """

    def __init__(self, root: str | Path, augment: bool = True, seed: int = 0) -> None:
        self.root = Path(root)
        self.augment = augment
        self._rng = random.Random(seed)

        img_dir = self.root / "training" / "rgb"
        if not img_dir.exists():
            raise FileNotFoundError(f"FreiHAND images not found at {img_dir}")

        self.image_paths = sorted(img_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images in {img_dir}")

        xyz_path = self.root / "training_xyz.json"
        K_path = self.root / "training_K.json"
        if not xyz_path.exists():
            raise FileNotFoundError(f"Missing {xyz_path}")
        if not K_path.exists():
            raise FileNotFoundError(f"Missing {K_path}")

        with open(xyz_path) as f:
            self._xyz = json.load(f)  # list of 32560 entries, each 21x3
        with open(K_path) as f:
            self._K = json.load(f)    # list of 32560 entries, each 3x3

        self._n_unique = len(self._xyz)  # 32560
        print(f"FreiHAND: {len(self.image_paths)} images, "
              f"{self._n_unique} unique annotations")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.image_paths[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")

        # 4 augmented versions share same annotation (idx % n_unique)
        ann_idx = idx % self._n_unique
        xyz = np.array(self._xyz[ann_idx], dtype=np.float32)  # [21, 3]
        K = np.array(self._K[ann_idx], dtype=np.float32)      # [3, 3]

        kp_2d = _project_3d_to_2d(xyz, K)  # [21, 2] in original image pixels

        lb_img, lb = letterbox_image(img, INPUT_SIZE)
        kp_lb = kp_2d.copy()
        kp_lb[:, 0] = kp_2d[:, 0] * lb.scale + lb.pad_x
        kp_lb[:, 1] = kp_2d[:, 1] * lb.scale + lb.pad_y
        kp_lb = np.clip(kp_lb, 0, INPUT_SIZE - 1).astype(np.float32)

        if self.augment:
            lb_img, kp_lb = _augment_sample(lb_img, kp_lb, self._rng)

        inp = normalize_rtmpose(lb_img)
        target = torch.from_numpy(kp_lb)
        return inp, target


class RHDDataset(Dataset):
    """RHD (Rendered Handpose Dataset) native labels.

    Each RHD image contains two hands (left [0:21] and right [21:42]).
    We extract each visible hand as a separate sample, cropped and
    letterboxed to INPUT_SIZE.

    Args:
        root: RHD root path containing ``{training,evaluation}/color/`` and
              ``anno_{training,evaluation}.mat``.
        splits: Which splits to use.
        augment: Enable training augmentations.
    """

    def __init__(
        self,
        root: str | Path,
        splits: Sequence[str] = ("training",),
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.augment = augment
        self._rng = random.Random(seed)

        self.samples: list[tuple[Path, np.ndarray]] = []
        self._load_splits(splits)
        print(f"RHD: {len(self.samples)} hand samples from {list(splits)}")

    def _load_splits(self, splits: Sequence[str]) -> None:
        for split in splits:
            # RHD stores annotations in multiple possible locations:
            #   root/anno_{split}.pickle          (flat layout)
            #   root/{split}/anno_{split}.pickle  (inside split dir)
            #   root/RHD_published_v2/{split}/anno_{split}.pickle  (nested zip)
            search_dirs = [self.root, self.root / split]
            for child in self.root.iterdir():
                if child.is_dir():
                    search_dirs.append(child)
                    search_dirs.append(child / split)

            found = False
            for d in search_dirs:
                if not d.is_dir():
                    continue
                for suffix in (".pickle", ".mat"):
                    p = d / f"anno_{split}{suffix}"
                    if p.exists():
                        base = d if (d / "color").is_dir() else d.parent if (d.parent / split / "color").is_dir() else d
                        if suffix == ".pickle":
                            self._load_pickle(split, p, base)
                        else:
                            self._load_mat(split, p, base)
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"WARNING: RHD anno not found for split '{split}' under {self.root}")

    def _find_color_dir(self, split: str, anno_parent: Path | None = None) -> Path | None:
        """Find the color image directory for a split."""
        candidates = []
        if anno_parent:
            candidates.append(anno_parent)
            if anno_parent.parent != self.root:
                candidates.append(anno_parent.parent)
        candidates.append(self.root)
        for child in self.root.iterdir():
            if child.is_dir():
                candidates.append(child)
        for base in candidates:
            d = base / split / "color"
            if d.is_dir():
                return d
            d2 = base / "color"
            if d2.is_dir():
                return d2
        return None

    def _load_pickle(self, split: str, path: Path, base_dir: Path | None = None) -> None:
        import pickle
        with open(path, "rb") as f:
            annos = pickle.load(f)

        color_dir = self._find_color_dir(split, base_dir or path.parent)
        if color_dir is None:
            print(f"WARNING: RHD color dir not found for split '{split}'")
            return

        for frame_idx, anno in annos.items():
            img_path = color_dir / f"{frame_idx:05d}.png"
            if not img_path.exists():
                continue
            uv_vis = np.array(anno["uv_vis"], dtype=np.float32)  # [42, 3]
            self._extract_hands(img_path, uv_vis)

    def _load_mat(self, split: str, path: Path, base_dir: Path | None = None) -> None:
        import scipy.io
        mat = scipy.io.loadmat(str(path))

        color_dir = self._find_color_dir(split, base_dir or path.parent)
        if color_dir is None:
            print(f"WARNING: RHD color dir not found for split '{split}'")
            return

        if "uv_vis" in mat:
            uv_vis_all = mat["uv_vis"]
        else:
            key = [k for k in mat.keys() if not k.startswith("_")]
            if not key:
                print(f"WARNING: No usable keys in {path}")
                return
            uv_vis_all = mat[key[0]]

        for i in range(uv_vis_all.shape[0]):
            img_path = color_dir / f"{i:05d}.png"
            if not img_path.exists():
                continue
            uv_vis = uv_vis_all[i].astype(np.float32)  # [42, 3]
            if uv_vis.ndim == 1:
                uv_vis = uv_vis.reshape(42, 3)
            self._extract_hands(img_path, uv_vis)

    def _extract_hands(self, img_path: Path, uv_vis: np.ndarray) -> None:
        """Extract left and right hands from 42-joint annotation if visible."""
        for hand_offset in (0, 21):
            hand_uv = uv_vis[hand_offset:hand_offset + 21]  # [21, 3]
            vis = hand_uv[:, 2]  # visibility flags
            if vis.sum() < 5:
                continue

            kp_xy = hand_uv[:, :2].copy()  # [21, 2] pixel coords in 320x320
            self.samples.append((img_path, kp_xy))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, kp_orig = self.samples[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")

        lb_img, lb = letterbox_image(img, INPUT_SIZE)
        kp_lb = kp_orig.copy()
        kp_lb[:, 0] = kp_orig[:, 0] * lb.scale + lb.pad_x
        kp_lb[:, 1] = kp_orig[:, 1] * lb.scale + lb.pad_y
        kp_lb = np.clip(kp_lb, 0, INPUT_SIZE - 1).astype(np.float32)

        if self.augment:
            lb_img, kp_lb = _augment_sample(lb_img, kp_lb, self._rng)

        inp = normalize_rtmpose(lb_img)
        target = torch.from_numpy(kp_lb)
        return inp, target


def build_native_dataset(
    freihand_root: str | Path | None = None,
    rhd_root: str | Path | None = None,
    augment: bool = True,
    seed: int = 0,
) -> Dataset:
    """Build a combined FreiHAND + RHD dataset from native labels.

    Either or both dataset paths can be provided.
    """
    datasets = []

    if freihand_root is not None:
        datasets.append(FreiHANDDataset(freihand_root, augment=augment, seed=seed))
    if rhd_root is not None:
        rhd_root = Path(rhd_root)
        actual_root = rhd_root
        for child in rhd_root.iterdir():
            if child.is_dir() and "rhd" in child.name.lower():
                if (child / "training" / "color").is_dir():
                    actual_root = child
                    break
        datasets.append(RHDDataset(actual_root, splits=("training", "evaluation"),
                                   augment=augment, seed=seed))

    if not datasets:
        raise ValueError("At least one of freihand_root or rhd_root must be provided")

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freihand", type=str, default=None)
    ap.add_argument("--rhd", type=str, default=None)
    args = ap.parse_args()

    ds = build_native_dataset(args.freihand, args.rhd, augment=False)
    print(f"Total samples: {len(ds)}")

    inp, tgt = ds[0]
    print(f"Sample 0: input={inp.shape}, target={tgt.shape}")
    print(f"  Input range: [{inp.min():.2f}, {inp.max():.2f}]")
    print(f"  Target range: [{tgt.min():.1f}, {tgt.max():.1f}]")
