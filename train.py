"""
train.py

Full training pipeline for the KLA semiconductor SR/denoising challenge.
Integrates custom loss modules, modular configurations, and model architectures.

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import make_train_val_split
from models import build_model
from losses import RestorationLoss
from metrics import compute_psnr, compute_ssim
from utils import load_config, set_seed


def get_device() -> torch.device:
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

            # Gradient clipping norm from config
            if hasattr(scaler, "unscale_"):
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr)
            loss = criterion(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        if use_amp:
            with autocast(device_type=device.type):
                pred = model(lr)
                loss = criterion(pred, gt)
        else:
            pred = model(lr)
            loss = criterion(pred, gt)

        # Dynamic metric tracking utilizing local bounds
        data_range = gt.amax(dim=[1, 2, 3]) - gt.amin(dim=[1, 2, 3])
        data_range = torch.clamp(data_range, min=1e-6)

        psnr = compute_psnr(pred, gt, data_range)
        ssim = compute_ssim(pred, gt, data_range)

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
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--root", type=str, default=None, help="Override dataset root path if provided")
    args = parser.parse_args()

    # --- Setup Config and Environment ---
    cfg = load_config(args.config)
    if args.root:
        cfg["data"]["root"] = args.root

    set_seed(cfg["train"].get("seed", 42))

    device = get_device()
    print(f"Using device: {device}")

    use_amp = device.type == "cuda" and cfg["train"].get("amp", True)
    print(f"Mixed precision (AMP): {'Enabled' if use_amp else 'Disabled'}")

    # --- Data Pipeline Setup ---
    train_ds, val_ds = make_train_val_split(
        root=cfg["data"]["root"],
        scale=cfg["data"].get("scale", 2),
        extensions=tuple(cfg["data"].get("extensions", [".npy", ".png", ".jpg", ".jpeg"])),
        val_fraction=cfg["data"].get("val_fraction", 0.1),
        patch_size=cfg["data"].get("patch_size", 96),
        augment_cfg=cfg.get("augmentation", None),
        seed=cfg["train"].get("seed", 42),
    )
    print(f"Dataset Loaded. Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and (device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and (device.type == "cuda"),
    )

    # --- Engine Instantiation ---
    model = build_model(cfg).to(device)

    # Enable Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU detected! Utilizing {torch.cuda.device_count()} GPUs via DataParallel.")
        model = torch.nn.DataParallel(model)

    criterion = RestorationLoss(cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"])
    )

    # Warmup + Cosine Scheduling Setup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=float(cfg["train"].get("min_lr", 1e-6))
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_psnr = -float("inf")

    # --- Resume Execution Handling ---
    resume_path = cfg["train"].get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming pipeline execution from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -float("inf"))

    checkpoint_dir = cfg["train"].get("checkpoint_dir", "./checkpoints")

    # --- Engine Execution Loop ---
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{cfg['train']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f}dB | Val SSIM: {val_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        # Safely extract state dict whether using DataParallel or single GPU
        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        meta_state = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "cfg": cfg,
        }

        # Maintain ongoing fallbacks for sudden execution limits
        save_checkpoint(meta_state, os.path.join(checkpoint_dir, "last.pt"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            meta_state["best_psnr"] = best_psnr
            save_checkpoint(meta_state, os.path.join(checkpoint_dir, "best.pt"))
            print(f"  -> Target metric improved! Model saved (PSNR={best_psnr:.2f}dB)")


if __name__ == "__main__":
    main()
