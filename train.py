"""
train.py

Full training pipeline for the KLA semiconductor SR/denoising challenge.

Usage:
    python train.py --root /path/to/train --epochs 100 --batch_size 8

Before running this, run verify_dims.py once to confirm the scale factor
matches the `scale` argument used to build the model below.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import make_train_val_split
from model import SRResidualUNet
from metrics import compute_psnr, compute_ssim, denormalize


def get_device() -> torch.device:
    """
    Never call .cuda() directly -- it raises AssertionError on CPU-only
    installs. Always resolve the device dynamically like this.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    running_loss = 0.0

    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type=device.type):
                pred = model(lr)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * lr.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    n_batches = 0

    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        gt_mean = batch["gt_mean"].to(device, non_blocking=True).float()
        gt_std = batch["gt_std"].to(device, non_blocking=True).float()

        if use_amp:
            with autocast(device_type=device.type):
                pred = model(lr)
                loss = criterion(pred, gt)
        else:
            pred = model(lr)
            loss = criterion(pred, gt)

        # De-normalize both pred and gt back to pixel space for PSNR/SSIM,
        # since metrics computed on z-scored values aren't meaningful.
        pred_denorm = denormalize(pred.float(), gt_mean, gt_std)
        gt_denorm = denormalize(gt.float(), gt_mean, gt_std)

        data_range = gt_denorm.amax(dim=[1, 2, 3]) - gt_denorm.amin(dim=[1, 2, 3])
        data_range = torch.clamp(data_range, min=1e-6)

        psnr = compute_psnr(pred_denorm, gt_denorm, data_range)
        ssim = compute_ssim(pred_denorm, gt_denorm, data_range)

        running_loss += loss.item()
        running_psnr += psnr.item()
        running_ssim += ssim.item()
        n_batches += 1

    return (
        running_loss / n_batches,
        running_psnr / n_batches,
        running_ssim / n_batches,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to train/ directory containing GT/ and NoisyLR/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--scale", type=int, default=2,
                         help="Super-resolution factor; confirm with verify_dims.py")
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    use_amp = device.type == "cuda"
    if use_amp:
        print("Mixed precision (AMP) enabled.")
    else:
        print("AMP disabled (CPU-only environment). "
              "Install a CUDA-enabled PyTorch build to enable it.")

    # --- Data ---
    train_ds, val_ds = make_train_val_split(args.root, val_fraction=args.val_fraction)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = SRResidualUNet(
        in_channels=1, out_channels=1, base_ch=args.base_ch, scale=args.scale
    ).to(device)

    # --- Loss / Optimizer / Scheduler ---
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_psnr = -float("inf")

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -float("inf"))

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, use_amp
        )

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_PSNR={val_psnr:.2f}dB val_SSIM={val_ssim:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s"
        )

        # Always save the latest checkpoint (for resuming).
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "args": vars(args),
            },
            os.path.join(args.checkpoint_dir, "last.pt"),
        )

        # Save the best checkpoint separately, tracked by validation PSNR.
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint_dir, "best.pt"),
            )
            print(f"  -> New best model saved (PSNR={best_psnr:.2f}dB)")


if __name__ == "__main__":
    main()
