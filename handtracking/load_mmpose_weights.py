"""Download and convert mmpose RTMPose-M hand pre-trained weights to our model.

Usage (from repo root):
    python3 -m handtracking.load_mmpose_weights [--out checkpoints/rtmpose_hand_pretrained.pt]

The script:
1. Downloads the mmpose Hand5 checkpoint (~100 MB)
2. Maps ``backbone.*`` and ``head.*`` keys to our CSPNeXt + RTMCCHead layout
3. Saves a state_dict ready for ``RTMPoseHand.load_state_dict()``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

MMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
)


def download_checkpoint(url: str, cache_dir: str = "checkpoints") -> Path:
    """Download checkpoint if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    dst = Path(cache_dir) / fname
    if dst.exists():
        print(f"Using cached checkpoint: {dst}")
        return dst
    print(f"Downloading {url} ...")
    torch.hub.download_url_to_file(url, str(dst))
    print(f"Saved to {dst}")
    return dst


def map_mmpose_weights(mmpose_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map mmpose state_dict keys to our RTMPoseHand layout.

    mmpose uses:
        backbone.stem.0.conv.weight  -> our backbone.stem.0.conv.weight
        head.final_layer.weight      -> our head.final_layer.weight

    The mapping is mostly prefix substitution since our module names
    deliberately mirror mmpose's structure.
    """
    mapped = {}
    skipped = []

    for key, value in mmpose_sd.items():
        new_key = None

        if key.startswith("backbone."):
            new_key = key  # direct match
        elif key.startswith("head."):
            new_key = key  # direct match
        else:
            skipped.append(key)
            continue

        mapped[new_key] = value

    if skipped:
        print(f"Skipped {len(skipped)} keys not in backbone/head: {skipped[:5]}...")

    return mapped


def convert(checkpoint_path: str | Path, out_path: str | Path) -> None:
    """Load mmpose .pth, extract and map weights, save for our model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    mapped = map_mmpose_weights(sd)
    print(f"Mapped {len(mapped)} weight tensors")

    from handtracking.models.rtmpose_hand import RTMPoseHand
    model = RTMPoseHand()
    our_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped.keys())

    missing = our_keys - mapped_keys
    unexpected = mapped_keys - our_keys

    if missing:
        print(f"\nMissing in pretrained ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}")
    if unexpected:
        print(f"\nUnexpected in pretrained ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  {k}")

    result = model.load_state_dict(mapped, strict=False)
    if result.missing_keys:
        print(f"\nKeys initialised randomly: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"Unexpected keys (ignored): {len(result.unexpected_keys)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved converted weights to {out_path}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert mmpose RTMPose-M weights")
    ap.add_argument("--url", default=MMPOSE_URL, help="mmpose checkpoint URL")
    ap.add_argument("--pth", default=None, help="Local .pth path (skips download)")
    ap.add_argument("--out", default="checkpoints/rtmpose_hand_pretrained.pt",
                    help="Output path for converted state_dict")
    args = ap.parse_args()

    if args.pth:
        ckpt_path = Path(args.pth)
    else:
        ckpt_path = download_checkpoint(args.url)

    convert(ckpt_path, args.out)


if __name__ == "__main__":
    main()
