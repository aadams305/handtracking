"""
Training loop: cosine annealing LR with warmup, gradient clipping, EMA, checkpointing.
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

from handtracking.dataset import HandSimCCDataset
from handtracking.losses import SimCCGaussianSoftCELoss
from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax
from handtracking.qat_wrapper import QATSimCCWrapper, apply_qat_prepare


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * current
    After training, use ``ema.module`` for inference/export (smoother weights).
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
    """Evaluate MPJPE (px) on the loader using the model in eval mode.

    Returns dict with 'mpjpe', 'mpjpe_tips', per-joint errors 'j0'..'j20', and 'max_joint_err'.
    """
    from handtracking.losses import FINGERTIP_INDICES

    model.eval()
    errs_sum = None
    n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x)
        lx, ly = out[0], out[1]
        pred = decode_simcc_soft_argmax(lx, ly)  # [B, J, 2]
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
    loss_fn: SimCCGaussianSoftCELoss,
    opt: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    ema: ModelEMA | None = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
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

        out = model(x)
        if len(out) == 4:
            lx, ly, pres_logit, hand_logit = out
            loss = loss_fn(lx, ly, y, pres_logit, hand_logit, has_hand, hand_label)
        else:
            lx, ly = out
            loss = loss_fn(lx, ly, y)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    ap.add_argument("--coord-loss-weight", type=float, default=1.0, help="Weight for coordinate L1 auxiliary loss")
    ap.add_argument("--sigma-bins", type=float, default=0.75, help="Gaussian sigma for bin targets (0.75 = tighter supervision)")
    ap.add_argument("--qat", action="store_true", help="QAT for last --qat-epochs")
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--width-mult", type=float, default=0.75)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (0 to disable)")
    ap.add_argument("--eval-every", type=int, default=5, help="Compute MPJPE metrics every N epochs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    ds = HandSimCCDataset(args.manifest, augment=True)
    if len(ds) == 0:
        raise SystemExit("Empty manifest; run distill_freihand first.")
    ds_eval = HandSimCCDataset(args.manifest, augment=False)
    print(f"Dataset: {len(ds)} samples", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    eval_loader = DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    base = HandSimCCNet(width_mult=args.width_mult).to(device)
    loss_fn = SimCCGaussianSoftCELoss(
        sigma_bins=args.sigma_bins,
        coord_loss_weight=args.coord_loss_weight,
    ).to(device)

    n_params = sum(p.numel() for p in base.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    fp_epochs = args.epochs - args.qat_epochs if args.qat else args.epochs
    if fp_epochs < 0:
        raise SystemExit("--epochs must be >= --qat-epochs when --qat")

    opt = AdamW(base.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineWarmupScheduler(opt, warmup_epochs=args.warmup_epochs, total_epochs=fp_epochs)

    ema: ModelEMA | None = None
    if args.ema_decay > 0:
        ema = ModelEMA(base, decay=args.ema_decay)
        print(f"EMA enabled (decay={args.ema_decay})", flush=True)

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
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        print(f"Resumed at epoch {start_epoch}", flush=True)

    best_loss = float("inf")
    best_mpjpe = float("inf")
    t_start = time.time()

    for epoch in range(start_epoch, fp_epochs):
        t0 = time.time()
        loss = train_epoch(base, loader, loss_fn, opt, device, grad_clip=args.grad_clip, ema=ema)
        sched.step()
        lr = sched.get_last_lr()[0]
        dt = time.time() - t0
        marker = " *best*" if loss < best_loss else ""
        log_line = f"epoch {epoch+1}/{args.epochs} (fp32) loss={loss:.6f} lr={lr:.6f} time={dt:.1f}s{marker}"

        # Periodic MPJPE evaluation
        eval_metrics: dict[str, float] | None = None
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == fp_epochs):
            eval_model = ema.module if ema is not None else base
            eval_metrics = compute_mpjpe(eval_model, eval_loader, device)
            mpjpe_str = f"  MPJPE={eval_metrics['mpjpe']:.2f}px tips={eval_metrics['mpjpe_tips']:.2f}px worst_j={eval_metrics['max_joint_err']:.2f}px"
            log_line += mpjpe_str
            if eval_metrics["mpjpe"] < best_mpjpe:
                best_mpjpe = eval_metrics["mpjpe"]
                log_line += " *best_mpjpe*"

        print(log_line, flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model": base.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "width_mult": args.width_mult,
            "qat": False,
        }
        if ema is not None:
            save_dict["ema"] = ema.state_dict()
        torch.save(save_dict, latest_path)

        if loss < best_loss:
            best_loss = loss
            best_path = args.out.with_name(args.out.stem + "_best.pt")
            save_best = {
                "model": (ema.module if ema else base).state_dict(),
                "width_mult": args.width_mult,
                "qat": False,
                "loss": loss,
            }
            if eval_metrics:
                save_best["mpjpe"] = eval_metrics["mpjpe"]
            torch.save(save_best, best_path)

    total_time = time.time() - t_start
    print(f"Training complete in {total_time/60:.1f} min. Best loss={best_loss:.6f}, best MPJPE={best_mpjpe:.2f}px", flush=True)

    # For final export, prefer EMA weights
    if ema is not None:
        model: nn.Module = ema.module.cpu()
    else:
        model = base.cpu()

    if args.qat and args.qat_epochs > 0:
        fp32_path = args.out.with_suffix(".fp32.pt")
        torch.save(
            {"model": model.state_dict(), "width_mult": args.width_mult, "qat": False},
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "width_mult": args.width_mult, "qat": bool(args.qat)},
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
