"""
Training loop: exponential LR decay, optional QAT, checkpointing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from handtracking.dataset import HandSimCCDataset
from handtracking.losses import SimCCGaussianSoftCELoss
from handtracking.models.hand_simcc import HandSimCCNet
from handtracking.qat_wrapper import QATSimCCWrapper, apply_qat_prepare


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SimCCGaussianSoftCELoss,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        lx, ly = model(x)
        loss = loss_fn(lx, ly, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-gamma", type=float, default=0.95, help="ExponentialLR gamma per epoch")
    ap.add_argument("--qat", action="store_true", help="QAT for last --qat-epochs")
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--width-mult", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = HandSimCCDataset(args.manifest, augment=True)
    if len(ds) == 0:
        raise SystemExit("Empty manifest; run distill_freihand first.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    base = HandSimCCNet(width_mult=args.width_mult).to(device)
    loss_fn = SimCCGaussianSoftCELoss().to(device)

    fp_epochs = args.epochs - args.qat_epochs if args.qat else args.epochs
    if fp_epochs < 0:
        raise SystemExit("--epochs must be >= --qat-epochs when --qat")

    opt = AdamW(base.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = ExponentialLR(opt, gamma=args.lr_gamma)

    latest_path = args.out.with_name(args.out.stem + "_latest.pt")
    start_epoch = 0

    if args.resume and latest_path.exists():
        print(f"Resuming from {latest_path} ...")
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest_path, map_location=device)
        base.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, fp_epochs):
        loss = train_epoch(base, loader, loss_fn, opt, device)
        sched.step()
        print(f"epoch {epoch+1}/{args.epochs} (fp32) loss={loss:.6f} lr={sched.get_last_lr()[0]:.6f}")
        
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": base.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "width_mult": args.width_mult,
                "qat": False
            },
            latest_path,
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
        sched = ExponentialLR(opt, gamma=args.lr_gamma)
        for epoch in range(args.qat_epochs):
            loss = train_epoch(model, loader, loss_fn, opt, device)
            sched.step()
            print(
                f"epoch {fp_epochs + epoch + 1}/{args.epochs} (qat) loss={loss:.6f} lr={sched.get_last_lr()[0]:.6f}"
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
