"""Training loop for RTMPose-M hand landmark model.

Features: cosine annealing LR with warmup, gradient clipping, EMA,
differential learning rates (backbone vs head), KLDiscretLoss,
MPJPE evaluation, checkpointing with resume support.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from handtracking.dataset_native import build_native_dataset
from handtracking.losses import KLDiscretLoss
from handtracking.models.rtmpose_hand import (
    INPUT_SIZE,
    NUM_BINS,
    SIMCC_SPLIT_RATIO,
    RTMPoseHand,
    decode_simcc,
)


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy updated as:  shadow = decay * shadow + (1 - decay) * current
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(model_p.data, alpha=1.0 - d)
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "decay": self.decay}

    def load_state_dict(self, d: dict) -> None:
        self.module.load_state_dict(d["module"])
        self.decay = d.get("decay", self.decay)


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


@torch.no_grad()
def compute_mpjpe(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate MPJPE (px) using RTMPose 512-bin SimCC decode."""
    from handtracking.losses import FINGERTIP_INDICES

    model.eval()
    errs_sum = None
    n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred_x, pred_y = model(x)
        pred = decode_simcc(pred_x, pred_y)  # [B, J, 2]
        per_joint = (pred - y).norm(dim=-1)  # [B, J]
        if errs_sum is None:
            errs_sum = per_joint.sum(dim=0)
        else:
            errs_sum += per_joint.sum(dim=0)
        n += x.size(0)
    model.train()
    if errs_sum is None or n == 0:
        return {"mpjpe": float("inf")}
    mean_per_joint = errs_sum / n
    mpjpe = float(mean_per_joint.mean().item())
    tip_errs = [float(mean_per_joint[i].item()) for i in FINGERTIP_INDICES if i < len(mean_per_joint)]
    result: dict[str, float] = {
        "mpjpe": mpjpe,
        "mpjpe_tips": sum(tip_errs) / max(1, len(tip_errs)),
        "max_joint_err": float(mean_per_joint.max().item()),
    }
    for j in range(len(mean_per_joint)):
        result[f"j{j}"] = float(mean_per_joint[j].item())
    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: KLDiscretLoss,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        opt.zero_grad(set_to_none=True)

        pred_x, pred_y = model(x)
        loss = loss_fn(pred_x, pred_y, y)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if ema is not None:
            ema.update(model)

        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train RTMPose-M hand landmark model")
    # Data
    ap.add_argument("--freihand", type=str, default=None, help="FreiHAND dataset root")
    ap.add_argument("--rhd", type=str, default=None, help="RHD dataset root")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Legacy JSONL manifest (for old MobileNetV4 pipeline)")
    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4, help="Head learning rate")
    ap.add_argument("--backbone-lr-scale", type=float, default=0.1,
                    help="Backbone LR = lr * backbone_lr_scale (differential LR)")
    ap.add_argument("--warmup-epochs", type=int, default=10, help="Linear LR warmup epochs")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    # Loss
    ap.add_argument("--sigma", type=float, default=6.0, help="Gaussian sigma for KL soft targets")
    ap.add_argument("--coord-loss-weight", type=float, default=1.0)
    # Model
    ap.add_argument("--pretrained", type=str, default=None,
                    help="Path to converted mmpose pretrained weights (.pt)")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/rtmpose_hand.pt"))
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    # Infra
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (0 to disable)")
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- Dataset ---
    if args.freihand or args.rhd:
        ds = build_native_dataset(args.freihand, args.rhd, augment=True)
        ds_eval = build_native_dataset(args.freihand, args.rhd, augment=False)
    elif args.manifest:
        from handtracking.dataset import HandSimCCDataset
        ds = HandSimCCDataset(args.manifest, augment=True)
        ds_eval = HandSimCCDataset(args.manifest, augment=False)
    else:
        raise SystemExit("Provide --freihand/--rhd for native labels, or --manifest for legacy.")

    if len(ds) == 0:
        raise SystemExit("Empty dataset.")
    print(f"Dataset: {len(ds)} training samples", flush=True)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    eval_loader = DataLoader(
        ds_eval, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )

    # --- Model ---
    model = RTMPoseHand().to(device)
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location=device, weights_only=True)
        result = model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained: missing={len(result.missing_keys)}, "
              f"unexpected={len(result.unexpected_keys)}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # --- Loss ---
    loss_fn = KLDiscretLoss(
        input_size=INPUT_SIZE,
        num_bins=NUM_BINS,
        split_ratio=SIMCC_SPLIT_RATIO,
        sigma=args.sigma,
        coord_loss_weight=args.coord_loss_weight,
    ).to(device)

    # --- Optimizer with differential LR ---
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    backbone_lr = args.lr * args.backbone_lr_scale
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
        {"params": head_params, "lr": args.lr, "name": "head"},
    ]
    opt = AdamW(param_groups, weight_decay=args.weight_decay)
    sched = CosineWarmupScheduler(opt, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
    print(f"Optimizer: backbone_lr={backbone_lr:.2e}, head_lr={args.lr:.2e}", flush=True)

    # --- EMA ---
    ema: ModelEMA | None = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"EMA enabled (decay={args.ema_decay})", flush=True)

    # --- Resume ---
    latest_path = args.out.with_name(args.out.stem + "_latest.pt")
    start_epoch = 0
    if args.resume and latest_path.exists():
        print(f"Resuming from {latest_path} ...", flush=True)
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        print(f"Resumed at epoch {start_epoch}", flush=True)

    # --- Training ---
    best_loss = float("inf")
    best_mpjpe = float("inf")
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_epoch(model, loader, loss_fn, opt, device, grad_clip=args.grad_clip, ema=ema)
        sched.step()
        lrs = sched.get_last_lr()
        dt = time.time() - t0
        marker = " *best*" if loss < best_loss else ""
        log_line = (f"epoch {epoch+1}/{args.epochs} loss={loss:.6f} "
                    f"bb_lr={lrs[0]:.2e} head_lr={lrs[1]:.2e} time={dt:.1f}s{marker}")

        eval_metrics: dict[str, float] | None = None
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            eval_model = ema.module if ema is not None else model
            eval_metrics = compute_mpjpe(eval_model, eval_loader, device)
            log_line += (f"  MPJPE={eval_metrics['mpjpe']:.2f}px "
                         f"tips={eval_metrics['mpjpe_tips']:.2f}px "
                         f"worst={eval_metrics['max_joint_err']:.2f}px")
            if eval_metrics["mpjpe"] < best_mpjpe:
                best_mpjpe = eval_metrics["mpjpe"]
                log_line += " *best_mpjpe*"

        print(log_line, flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "arch": "rtmpose_hand",
        }
        if ema is not None:
            save_dict["ema"] = ema.state_dict()
        torch.save(save_dict, latest_path)

        if loss < best_loss:
            best_loss = loss
            best_path = args.out.with_name(args.out.stem + "_best.pt")
            save_best = {
                "model": (ema.module if ema else model).state_dict(),
                "arch": "rtmpose_hand",
                "loss": loss,
            }
            if eval_metrics:
                save_best["mpjpe"] = eval_metrics["mpjpe"]
            torch.save(save_best, best_path)

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} min", flush=True)
    print(f"Best loss={best_loss:.6f}, best MPJPE={best_mpjpe:.2f}px", flush=True)

    final_model = (ema.module if ema else model).cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": final_model.state_dict(), "arch": "rtmpose_hand"},
        args.out,
    )
    print(f"Saved final model: {args.out}")


if __name__ == "__main__":
    main()
