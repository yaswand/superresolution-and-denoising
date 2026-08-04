"""
infer_visualize.py

Loads a checkpoint (best.pt) and runs inference on validation images,
saving side-by-side plots: NoisyLR (upsampled for display) | Prediction | GT.

Usage:
    python infer_visualize.py --root /path/to/train --checkpoint ./checkpoints/best.pt --n_samples 8
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import make_train_val_split
from model import SRResidualUNet
from metrics import compute_psnr, compute_ssim, denormalize


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """[1, H, W] tensor -> [H, W] numpy for imshow."""
    return t.squeeze(0).cpu().numpy()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to train/ directory (same one used for training)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="./val_visualizations")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Rebuild the SAME train/val split used during training (same seed in
    # make_train_val_split), so these are genuinely held-out validation images.
    _, val_ds = make_train_val_split(args.root, val_fraction=args.val_fraction)
    print(f"Validation set size: {len(val_ds)}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    scale = ckpt_args.get("scale", 2)
    base_ch = ckpt_args.get("base_ch", 64)

    model = SRResidualUNet(in_channels=1, out_channels=1,
                            base_ch=base_ch, scale=scale).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(best_psnr={ckpt.get('best_psnr', 'N/A')})")

    os.makedirs(args.out_dir, exist_ok=True)

    n_samples = min(args.n_samples, len(val_ds))
    psnr_list, ssim_list = [], []

    for i in range(n_samples):
        sample = val_ds[i]
        lr = sample["lr"].unsqueeze(0).to(device)   # [1, 1, H, W]
        gt = sample["gt"].unsqueeze(0).to(device)    # [1, 1, 2H, 2W]
        gt_mean = torch.tensor([sample["gt_mean"]], device=device).float()
        gt_std = torch.tensor([sample["gt_std"]], device=device).float()
        lr_mean = torch.tensor([sample["lr_mean"]], device=device).float()
        lr_std = torch.tensor([sample["lr_std"]], device=device).float()

        pred = model(lr)

        # De-normalize everything back to pixel space for display + metrics.
        pred_denorm = denormalize(pred, gt_mean, gt_std)
        gt_denorm = denormalize(gt, gt_mean, gt_std)
        lr_denorm = denormalize(lr, lr_mean, lr_std)

        data_range = gt_denorm.amax(dim=[1, 2, 3]) - gt_denorm.amin(dim=[1, 2, 3])
        data_range = torch.clamp(data_range, min=1e-6)

        psnr = compute_psnr(pred_denorm, gt_denorm, data_range).item()
        ssim = compute_ssim(pred_denorm, gt_denorm, data_range).item()
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        # Upsample LR (nearest, no smoothing) just for side-by-side display
        # at the same resolution as pred/GT -- this is display-only, not
        # used anywhere in training or metrics.
        lr_display = F.interpolate(lr_denorm, size=gt_denorm.shape[-2:],
                                    mode="nearest")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(to_numpy_img(lr_display[0]), cmap="gray")
        axes[0].set_title("NoisyLR (upsampled for display)")
        axes[1].imshow(to_numpy_img(pred_denorm[0]), cmap="gray")
        axes[1].set_title(f"Prediction\nPSNR={psnr:.2f}dB SSIM={ssim:.3f}")
        axes[2].imshow(to_numpy_img(gt_denorm[0]), cmap="gray")
        axes[2].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")

        fname = sample["fname"].replace(".npy", ".png")
        out_path = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[{i + 1}/{n_samples}] {sample['fname']}: "
              f"PSNR={psnr:.2f}dB SSIM={ssim:.3f} -> saved {out_path}")

    print(f"\nMean over {n_samples} samples: "
          f"PSNR={np.mean(psnr_list):.2f}dB SSIM={np.mean(ssim_list):.3f}")
    print(f"All visualizations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
