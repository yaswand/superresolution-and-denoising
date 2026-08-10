"""
train_ddp.py

Full training pipeline for the KLA semiconductor SR/denoising challenge.
Integrates custom loss modules, modular configurations, and model architectures.

Usage:
    torchrun --standalone --nproc_per_node=2 train_ddp.py --config configs/default.yaml
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset import make_train_val_split
from models import build_model
from losses import RestorationLoss
from metrics import compute_psnr, compute_ssim
from utils import load_config, set_seed
from evaluate import run_full_evaluation


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int]:
    """
    Initialize torch.distributed if the script is launched with torchrun.

    Returns:
        local_rank, world_size
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return local_rank, world_size


def cleanup_distributed():
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def reduce_stats(stats: torch.Tensor) -> torch.Tensor:
    """
    Sum-reduce stats across all ranks.
    """
    if is_dist_available_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp, distributed: bool = False):
    model.train()
    local_loss_sum = 0.0
    local_sample_count = 0

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

        batch_size = lr.size(0)
        local_loss_sum += loss.detach().item() * batch_size
        local_sample_count += batch_size

    stats = torch.tensor([local_loss_sum, local_sample_count], device=device, dtype=torch.float64)
    if distributed:
        reduce_stats(stats)

    total_loss_sum = stats[0].item()
    total_sample_count = max(int(stats[1].item()), 1)
    return total_loss_sum / total_sample_count


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, distributed: bool = False):
    model.eval()
    local_loss_sum = 0.0
    local_psnr_sum = 0.0
    local_ssim_sum = 0.0
    local_sample_count = 0

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

        batch_size = lr.size(0)
        local_loss_sum += loss.detach().item() * batch_size
        local_psnr_sum += psnr.detach().item() * batch_size
        local_ssim_sum += ssim.detach().item() * batch_size
        local_sample_count += batch_size

    stats = torch.tensor(
        [local_loss_sum, local_psnr_sum, local_ssim_sum, local_sample_count],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        reduce_stats(stats)

    total_sample_count = max(int(stats[3].item()), 1)
    return (
        stats[0].item() / total_sample_count,
        stats[1].item() / total_sample_count,
        stats[2].item() / total_sample_count,
    )


def build_eval_subset(dataset, rank: int, world_size: int):
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


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

    local_rank, world_size = setup_distributed()
    distributed = world_size > 1 and torch.cuda.is_available()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    if is_main_process():
        print(f"Using device: {device}")
        if distributed:
            print(f"Distributed training enabled. World size: {world_size} | Local rank: {local_rank}")

    use_amp = device.type == "cuda" and cfg["train"].get("amp", True)
    if is_main_process():
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

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=get_rank(), shuffle=True, drop_last=True
        )
        val_subset = build_eval_subset(val_ds, get_rank(), world_size)
        val_sampler = None
    else:
        train_sampler = None
        val_subset = val_ds
        val_sampler = None

    if is_main_process():
        print(f"Dataset Loaded. Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds if not distributed else train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and (device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and (device.type == "cuda"),
    )

    # --- Engine Instantiation ---
    model = build_model(cfg).to(device)

    if distributed:
        if is_main_process():
            print(f"Multi-GPU detected! Utilizing {world_size} GPUs via DistributedDataParallel.")
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    criterion = RestorationLoss(cfg)
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)

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
        if is_main_process():
            print(f"Resuming pipeline execution from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -float("inf"))

    if distributed:
        dist.barrier()

    checkpoint_dir = cfg["train"].get("checkpoint_dir", "./checkpoints")

    # --- Engine Execution Loop ---
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()

        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, distributed)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device, use_amp, distributed)

        scheduler.step()
        elapsed = time.time() - t0

        if is_main_process():
            print(
                f"Epoch [{epoch + 1}/{cfg['train']['epochs']}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val PSNR: {val_psnr:.2f}dB | Val SSIM: {val_ssim:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
            )

            # Safely extract state dict whether using DDP or single GPU
            model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

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

        if distributed:
            dist.barrier()

    # --- Post-training evaluation: ID + deterministic OOD, PSNR/SSIM/LPIPS ---
    # Runs once, on the main process only, single-device, against the best
    # checkpoint saved above. This does not change training itself (loss/
    # optimizer/scheduler/DDP are all already finished by this point) --
    # it just produces the reproducible eval artifact the ablation compares.
    if is_main_process():
        best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.isfile(best_ckpt_path):
            eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"\nRunning post-training ID + OOD evaluation on {best_ckpt_path} ...")
            try:
                results = run_full_evaluation(
                    checkpoint_path=best_ckpt_path,
                    config_path=None,  # use the cfg embedded in the checkpoint
                    device=eval_device,
                    lpips_net=cfg.get("metrics", {}).get("lpips_net", "alex"),
                    batch_size=cfg["train"]["batch_size"],
                    num_workers=cfg["data"]["num_workers"],
                )
                results_path = os.path.join(checkpoint_dir, "eval_best.json")
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved ID + OOD evaluation results to {results_path}")
            except Exception as e:
                # Evaluation is a reporting step; a failure here (e.g. missing
                # LPIPS weights / no network) must not be mistaken for a
                # training failure, so we log and continue rather than raise.
                print(f"[train_ddp] Post-training evaluation failed, skipping: {e}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
