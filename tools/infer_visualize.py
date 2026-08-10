"""
infer_visualize_current.py

Inference + visualization for the CURRENT codebase.

Assumed dataset:
    train/
      GT/
        *.npy
      NoisyLR/
        *.npy

Example:
    python infer_visualize_current.py \
        --root /kaggle/input/YOUR_DATASET/train \
        --n_samples 8 \
        --out_dir /kaggle/working/inference_results
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import _load_array
from metrics import compute_psnr, compute_ssim
from models import build_model
from utils import load_config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_pairs(root):
    gt_dir = os.path.join(root, "GT")
    lr_dir = os.path.join(root, "NoisyLR")

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"NoisyLR folder not found: {lr_dir}")

    gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)
                if f.lower().endswith(".npy")}
    lr_files = {os.path.splitext(f)[0]: f for f in os.listdir(lr_dir)
                if f.lower().endswith(".npy")}

    common = sorted(set(gt_files) & set(lr_files))
    if not common:
        raise RuntimeError(
            "No matching .npy files found between GT and NoisyLR. "
            "This script pairs files by filename stem."
        )

    return [
        (
            stem,
            os.path.join(lr_dir, lr_files[stem]),
            os.path.join(gt_dir, gt_files[stem]),
        )
        for stem in common
    ]


def as_tensor(arr, device):
    arr = np.asarray(arr, dtype=np.float32).squeeze()

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {arr.shape}")

    return (
        torch.from_numpy(np.ascontiguousarray(arr))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


@torch.no_grad()
def process_image(model, img_tensor, tile_size, overlap, scale):
    _, _, h, w = img_tensor.shape

    if tile_size is None or (h <= tile_size and w <= tile_size):
        return model(img_tensor)

    out_h, out_w = h * scale, w * scale
    out = torch.zeros(
        (1, 1, out_h, out_w),
        dtype=img_tensor.dtype,
        device=img_tensor.device,
    )
    count = torch.zeros_like(out)

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size")

    for i in range(0, h, stride):
        for j in range(0, w, stride):
            t = min(i + tile_size, h)
            r = min(j + tile_size, w)

            b = i if (t - i) == tile_size else max(0, t - tile_size)
            l = j if (r - j) == tile_size else max(0, r - tile_size)

            tile = img_tensor[:, :, b:t, l:r]
            pred_tile = model(tile)

            out[:, :, b * scale:t * scale, l * scale:r * scale] += pred_tile
            count[:, :, b * scale:t * scale, l * scale:r * scale] += 1

    return out / count.clamp_min(1)


def image_for_display(x):
    x = np.asarray(x).squeeze()
    lo, hi = np.percentile(x, [1, 99])

    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to train/ containing GT/ and NoisyLR/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_exp4patch128/best.pt",
        help="Checkpoint path (default: ./checkpoints/best.pt)",
    )
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="./inference_results")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when choosing samples",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Same checkpoint/model loading approach as the current infer.py.
    ckpt = torch.load(args.checkpoint, map_location=device)

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain 'model_state'. "
            "Pass the original best.pt produced by train.py."
        )

    cfg = ckpt.get("cfg")
    if cfg is None:
        print("Checkpoint has no cfg; falling back to configs/default.yaml")
        cfg = load_config("configs/default.yaml")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "N/A")
    best_psnr = ckpt.get("best_psnr", "N/A")
    print(f"Loaded checkpoint: epoch={epoch}, best_psnr={best_psnr}")

    inference_cfg = cfg.get("inference", {})
    use_amp = device.type == "cuda" and inference_cfg.get("amp", True)
    tile_size = inference_cfg.get("tile_size", None)
    overlap = inference_cfg.get("tile_overlap", 32)
    scale = int(cfg["model"]["scale"])

    pairs = find_pairs(args.root)
    print(f"Found {len(pairs)} matching GT/NoisyLR pairs")

    random.seed(args.seed)
    n = min(args.n_samples, len(pairs))
    chosen = random.sample(pairs, n) if n < len(pairs) else pairs

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "predictions")
    comparison_dir = os.path.join(args.out_dir, "comparisons")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    psnrs = []
    ssims = []

    for idx, (stem, lr_path, gt_path) in enumerate(chosen, 1):
        lr_np = _load_array(lr_path)
        gt_np = _load_array(gt_path)

        lr = as_tensor(lr_np, device)
        gt = as_tensor(gt_np, device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = process_image(
                model=model,
                img_tensor=lr,
                tile_size=tile_size,
                overlap=overlap,
                scale=scale,
            )

        if pred.shape[-2:] != gt.shape[-2:]:
            raise RuntimeError(
                f"{stem}: prediction shape {tuple(pred.shape[-2:])} "
                f"does not match GT shape {tuple(gt.shape[-2:])}"
            )

        # Metrics are computed on raw model/GT values, not display-normalized images.
        data_range = (
            gt.amax(dim=(1, 2, 3)) - gt.amin(dim=(1, 2, 3))
        ).clamp_min(1e-6)

        psnr = compute_psnr(pred.float(), gt.float(), data_range).item()
        ssim = compute_ssim(pred.float(), gt.float(), data_range).item()
        psnrs.append(psnr)
        ssims.append(ssim)

        pred_np = pred.squeeze().float().cpu().numpy()

        # Save raw prediction for later scoring/inspection.
        np.save(os.path.join(pred_dir, stem + ".npy"), pred_np)

        # Upsampling LR here is ONLY for visualization.
        lr_display_t = F.interpolate(
            lr.float(),
            size=gt.shape[-2:],
            mode="nearest",
        )

        lr_display = image_for_display(lr_display_t.squeeze().cpu().numpy())
        pred_display = image_for_display(pred_np)
        gt_display = image_for_display(gt.squeeze().cpu().numpy())

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].imshow(lr_display, cmap="gray")
        axes[0].set_title("NoisyLR\n(upsampled for display only)")

        axes[1].imshow(pred_display, cmap="gray")
        axes[1].set_title(
            f"Prediction\nPSNR={psnr:.2f} dB | SSIM={ssim:.4f}"
        )

        axes[2].imshow(gt_display, cmap="gray")
        axes[2].set_title("Ground Truth")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(stem)
        plt.tight_layout()

        comparison_path = os.path.join(comparison_dir, stem + ".png")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(
            f"[{idx}/{n}] {stem}: "
            f"PSNR={psnr:.2f} dB, SSIM={ssim:.4f}"
        )

    print("\nFinished.")
    print(f"Mean PSNR: {np.mean(psnrs):.2f} dB")
    print(f"Mean SSIM: {np.mean(ssims):.4f}")
    print(f"Raw predictions: {pred_dir}")
    print(f"Comparison images: {comparison_dir}")


if __name__ == "__main__":
    main()
