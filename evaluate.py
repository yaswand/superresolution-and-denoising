"""
evaluate.py

Standalone evaluation of a trained checkpoint on:
  - the existing in-distribution (ID) validation set (organizer split,
    same val_fraction/seed as training -> same stems every time), and
  - deterministic out-of-distribution (OOD) variants built by applying
    fixed extra perturbations on top of the *existing* validation NoisyLR
    inputs (see ood.py). GT is never modified and is always the original
    organizer-provided ground truth.

Reports PSNR, SSIM, and (if available) LPIPS for ID and for each OOD
variant, and writes everything to a JSON results file for reproducible,
checkpointed comparison across experiments (e.g. Experiment 1 vs
Experiment 3).

Usage:
    python evaluate.py --config configs/exp1_charbonnier.yaml \
        --checkpoint checkpoints_exp1/best.pt \
        --out results/exp1_eval.json

    python evaluate.py --config configs/exp3_no_lr_degradation.yaml \
        --checkpoint checkpoints_exp3/best.pt \
        --out results/exp3_eval.json

Notes:
  - This script does not require torch.distributed; it evaluates on a
    single device (GPU if available, else CPU). Training remains DDP;
    evaluation is intentionally single-process for simplicity and to
    guarantee a fixed, non-sharded iteration order over the val set.
  - LPIPS is optional. If the `lpips` package or its pretrained weights
    are unavailable, LPIPS is skipped (reported as null in the JSON) and
    a warning is printed; PSNR/SSIM are unaffected.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from dataset import make_train_val_split
from metrics import compute_psnr, compute_ssim
from models import build_model
from ood import apply_ood_variant, variant_names
from utils import load_config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, cfg_override: Optional[dict] = None):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = cfg_override if cfg_override is not None else ckpt.get("cfg")
    if cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no embedded 'cfg' and no --config was provided."
        )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt.get("epoch", None)


@torch.no_grad()
def _forward(model, lr: torch.Tensor, device: torch.device, use_amp: bool) -> torch.Tensor:
    lr = lr.to(device, non_blocking=True)
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type=device.type):
            pred = model(lr)
    else:
        pred = model(lr)
    return pred.float()


@torch.no_grad()
def evaluate_id(model, val_ds, device: torch.device, use_amp: bool, batch_size: int,
                 num_workers: int, lpips_metric=None) -> Dict[str, float]:
    """
    Standard ID evaluation: model(NoisyLR) vs GT, unmodified, in the
    existing validation split's natural iteration order (no shuffling).
    """
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    n = 0

    for batch in loader:
        lr = batch["lr"]
        gt = batch["gt"].to(device, non_blocking=True)

        pred = _forward(model, lr, device, use_amp)

        data_range = gt.amax(dim=[1, 2, 3]) - gt.amin(dim=[1, 2, 3])
        data_range = torch.clamp(data_range, min=1e-6)

        bsz = gt.size(0)
        psnr_sum += compute_psnr(pred, gt, data_range).item() * bsz
        ssim_sum += compute_ssim(pred, gt, data_range).item() * bsz

        if lpips_metric is not None:
            lpips_sum += lpips_metric(pred, gt) * bsz
            lpips_count += bsz

        n += bsz

    result = {
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
        "lpips": (lpips_sum / lpips_count) if lpips_count > 0 else None,
        "n_samples": n,
    }
    return result


@torch.no_grad()
def evaluate_ood(model, val_ds, variant: str, device: torch.device, use_amp: bool,
                  lpips_metric=None) -> Dict[str, float]:
    """
    OOD evaluation for a single variant: for each validation sample,
    deterministically perturbs the *existing* NoisyLR (see ood.py), runs
    the model, and compares against the original, untouched GT.

    Iterates one sample at a time (no batching) so that `sample_index`
    passed to apply_ood_variant is simply the position in the fixed val
    stem order -- keeping the seed derivation in ood.py trivial to reason
    about and independent of batch size / world size.
    """
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    n = 0

    for idx in range(len(val_ds)):
        sample = val_ds[idx]
        lr = sample["lr"]  # [1, H, W], original NoisyLR
        gt = sample["gt"].unsqueeze(0).to(device)  # [1, 1, H, W], untouched GT

        lr_ood = apply_ood_variant(lr, variant, sample_index=idx).unsqueeze(0)  # [1, 1, H, W]

        pred = _forward(model, lr_ood, device, use_amp)

        data_range = gt.amax(dim=[1, 2, 3]) - gt.amin(dim=[1, 2, 3])
        data_range = torch.clamp(data_range, min=1e-6)

        psnr_sum += compute_psnr(pred, gt, data_range).item()
        ssim_sum += compute_ssim(pred, gt, data_range).item()

        if lpips_metric is not None:
            lpips_sum += lpips_metric(pred, gt)
            lpips_count += 1

        n += 1

    return {
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
        "lpips": (lpips_sum / lpips_count) if lpips_count > 0 else None,
        "n_samples": n,
    }


def run_full_evaluation(
    checkpoint_path: str,
    config_path: Optional[str],
    device: torch.device,
    compute_lpips: bool,
    lpips_net: str,
    ood_variants: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    data_root: Optional[str] = None,
) -> dict:
    cfg_override = load_config(config_path) if config_path else None
    model, cfg, epoch = load_model_from_checkpoint(checkpoint_path, device, cfg_override)

    # Same override mechanism as train_ddp.py's --root: the dataset root in
    # the config (or in the checkpoint's embedded cfg) is very often a
    # relative path like "./train" that only resolves correctly from the
    # exact directory training was launched from. On a new environment
    # (e.g. a fresh Kaggle session) that directory won't exist, so let the
    # caller point explicitly at wherever the data actually landed this time.
    if data_root:
        cfg["data"]["root"] = data_root

    _, val_ds = make_train_val_split(
        root=cfg["data"]["root"],
        scale=cfg["data"].get("scale", 2),
        extensions=tuple(cfg["data"].get("extensions", [".npy", ".png", ".jpg", ".jpeg"])),
        val_fraction=cfg["data"].get("val_fraction", 0.1),
        patch_size=cfg["data"].get("patch_size", 96),
        augment_cfg=cfg.get("augmentation", None),
        seed=cfg["train"].get("seed", 42),
    )

    use_amp = device.type == "cuda" and cfg.get("inference", {}).get("amp", True)

    lpips_metric = None
    if compute_lpips:
        from lpips_metric import try_build_lpips

        lpips_metric = try_build_lpips(net=lpips_net, device=device)

    if ood_variants is None:
        ood_variants = variant_names()

    results = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "epoch": epoch,
        "n_val_samples": len(val_ds),
        "lpips_enabled": lpips_metric is not None,
        "lpips_net": lpips_net if lpips_metric is not None else None,
    }

    t0 = time.time()
    print(f"Evaluating ID validation set ({len(val_ds)} samples)...")
    results["id"] = evaluate_id(
        model, val_ds, device, use_amp, batch_size=batch_size, num_workers=num_workers, lpips_metric=lpips_metric
    )
    print(f"  ID: PSNR={results['id']['psnr']:.3f}dB SSIM={results['id']['ssim']:.4f} "
          f"LPIPS={results['id']['lpips']}")

    results["ood"] = {}
    for variant in ood_variants:
        print(f"Evaluating OOD variant '{variant}' ({len(val_ds)} samples)...")
        r = evaluate_ood(model, val_ds, variant, device, use_amp, lpips_metric=lpips_metric)
        results["ood"][variant] = r
        print(f"  {variant}: PSNR={r['psnr']:.3f}dB SSIM={r['ssim']:.4f} LPIPS={r['lpips']}")

    results["elapsed_seconds"] = time.time() - t0
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on ID + deterministic OOD validation sets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint (e.g. best.pt).")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML to use for dataset/model construction. Defaults to the cfg embedded in the checkpoint.",
    )
    parser.add_argument("--out", type=str, default=None, help="Where to write the JSON results file.")
    parser.add_argument(
        "--root", type=str, default=None,
        help="Override data.root from the config / checkpoint's embedded cfg (e.g. the actual "
             "dataset path on this machine, such as /kaggle/input/<dataset>/train).",
    )
    parser.add_argument(
        "--lpips", dest="lpips", action="store_true", default=None,
        help="Force-enable LPIPS regardless of the config's metrics.compute_lpips.",
    )
    parser.add_argument(
        "--no-lpips", dest="lpips", action="store_false",
        help="Force-disable LPIPS regardless of the config's metrics.compute_lpips.",
    )
    parser.add_argument(
        "--ood-variants", type=str, nargs="*", default=None,
        help="Subset of OOD variants to run (default: all of ood.variant_names()).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Resolve LPIPS enable/disable: CLI flag overrides config; config default
    # (metrics.compute_lpips) is used if the flag is not passed at all.
    cfg_for_lpips_default = load_config(args.config) if args.config else None
    lpips_net = "alex"
    if cfg_for_lpips_default is not None:
        lpips_net = cfg_for_lpips_default.get("metrics", {}).get("lpips_net", "alex")
    if args.lpips is None:
        compute_lpips = (
            cfg_for_lpips_default.get("metrics", {}).get("compute_lpips", False)
            if cfg_for_lpips_default is not None
            else False
        )
    else:
        compute_lpips = args.lpips

    results = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        compute_lpips=compute_lpips,
        lpips_net=lpips_net,
        ood_variants=args.ood_variants,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.root,
    )

    out_path = args.out
    if out_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint) or "."
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(ckpt_dir, f"eval_{ckpt_name}.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
