"""
Training loop: cosine annealing LR with warmup, gradient clipping, checkpointing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from handtracking.dataset import HandSimCCDataset
from handtracking.losses import SimCCGaussianSoftCELoss
from handtracking.models.hand_simcc import HandSimCCNet
from handtracking.qat_wrapper import QATSimCCWrapper, apply_qat_prepare


class CosineWarmupScheduler:
    """Linear warmup + cosine annealing to eta_min."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._last_lr = list(self.base_lrs)
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
        epoch = self._step_count
        for i, (pg, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            if epoch <= self.warmup_epochs:
                lr = base_lr * epoch / max(1, self.warmup_epochs)
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr
            self._last_lr[i] = lr

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {"step_count": self._step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, d: dict) -> None:
        self._step_count = d["step_count"]
        self.base_lrs = d["base_lrs"]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SimCCGaussianSoftCELoss,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        # Dataset returns (image, keypoints, has_hand, handedness)
        if len(batch) == 4:
            x, y, has_hand, hand_label = batch
            has_hand = has_hand.to(device)
            hand_label = hand_label.to(device)
        else:
            x, y = batch
            has_hand = None
            hand_label = None

        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)

        # Model returns (lx, ly, presence_logit, handedness_logit) or just (lx, ly)
        out = model(x)
        if len(out) == 4:
            lx, ly, pres_logit, hand_logit = out
            loss = loss_fn(lx, ly, y, pres_logit, hand_logit, has_hand, hand_label)
        else:
            lx, ly = out
            loss = loss_fn(lx, ly, y)

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    ap.add_argument("--coord-loss-weight", type=float, default=0.5, help="Weight for coordinate L1 auxiliary loss")
    ap.add_argument("--sigma-bins", type=float, default=1.0, help="Gaussian sigma for bin targets")
    ap.add_argument("--qat", action="store_true", help="QAT for last --qat-epochs")
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--width-mult", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    ds = HandSimCCDataset(args.manifest, augment=True)
    if len(ds) == 0:
        raise SystemExit("Empty manifest; run distill_freihand first.")
    print(f"Dataset: {len(ds)} samples", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    base = HandSimCCNet(width_mult=args.width_mult).to(device)
    loss_fn = SimCCGaussianSoftCELoss(
        sigma_bins=args.sigma_bins,
        coord_loss_weight=args.coord_loss_weight,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in base.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    fp_epochs = args.epochs - args.qat_epochs if args.qat else args.epochs
    if fp_epochs < 0:
        raise SystemExit("--epochs must be >= --qat-epochs when --qat")

    opt = AdamW(base.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, warmup_epochs=args.warmup_epochs, total_epochs=fp_epochs)

    latest_path = args.out.with_name(args.out.stem + "_latest.pt")
    start_epoch = 0

    if args.resume and latest_path.exists():
        print(f"Resuming from {latest_path} ...", flush=True)
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest_path, map_location=device)
        base.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch}", flush=True)

    best_loss = float("inf")
    for epoch in range(start_epoch, fp_epochs):
        loss = train_epoch(base, loader, loss_fn, opt, device, grad_clip=args.grad_clip)
        sched.step()
        lr = sched.get_last_lr()[0]
        marker = " *best*" if loss < best_loss else ""
        print(f"epoch {epoch+1}/{args.epochs} (fp32) loss={loss:.6f} lr={lr:.6f}{marker}", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": base.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "width_mult": args.width_mult,
                "qat": False
            },
            latest_path,
        )

        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = args.out.with_name(args.out.stem + "_best.pt")
            torch.save(
                {"model": base.state_dict(), "width_mult": args.width_mult, "qat": False, "loss": loss},
                best_path,
            )

    model: nn.Module = base
    if args.qat and args.qat_epochs > 0:
        fp32_path = args.out.with_suffix(".fp32.pt")
        torch.save(
            {"model": base.state_dict(), "width_mult": args.width_mult, "qat": False},
            fp32_path,
        )
        print(f"saved FP32 weights for ONNX export: {fp32_path}")
        model = QATSimCCWrapper(base.cpu()).to(device)
        apply_qat_prepare(model)
        opt = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        sched_qat = CosineWarmupScheduler(opt, warmup_epochs=0, total_epochs=args.qat_epochs)
        for epoch in range(args.qat_epochs):
            loss = train_epoch(model, loader, loss_fn, opt, device, grad_clip=args.grad_clip)
            sched_qat.step()
            print(
                f"epoch {fp_epochs + epoch + 1}/{args.epochs} (qat) loss={loss:.6f} lr={sched_qat.get_last_lr()[0]:.6f}"
            )
        model.eval()
        model = torch.ao.quantization.convert(model.cpu(), inplace=False)
    else:
        model = base.cpu()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "width_mult": args.width_mult, "qat": bool(args.qat)},
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
