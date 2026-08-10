"""
verify_dims.py

Run this FIRST, before training. It inspects a sample of GT / NoisyLR pairs and
prints their shapes, dtypes, and value ranges. This determines:

  1. The scale factor the model must learn (1x vs 2x).
  2. Whether normalization stats should be computed per-image or globally.
  3. Whether speckle noise really does push NoisyLR values outside GT's range
     (confirming why clipping must be avoided).

Usage:
    python verify_dims.py --root /path/to/train
"""

import argparse
import os
import numpy as np


def inspect(root: str, n_samples: int = 20):
    gt_dir = os.path.join(root, "GT")
    lr_dir = os.path.join(root, "NoisyLR")

    files = sorted(os.listdir(gt_dir))[:n_samples]
    if not files:
        raise RuntimeError(f"No files found in {gt_dir}")

    print(f"{'file':<15} {'GT shape':<15} {'LR shape':<15} "
          f"{'GT range':<20} {'LR range':<20} scale")
    print("-" * 100)

    scales = set()
    for fname in files:
        gt = np.load(os.path.join(gt_dir, fname))
        lr = np.load(os.path.join(lr_dir, fname))

        gt_range = f"[{gt.min():.2f}, {gt.max():.2f}]"
        lr_range = f"[{lr.min():.2f}, {lr.max():.2f}]"

        # Assume square images; scale = GT height / LR height
        scale = gt.shape[0] / lr.shape[0]
        scales.add(scale)

        print(f"{fname:<15} {str(gt.shape):<15} {str(lr.shape):<15} "
              f"{gt_range:<20} {lr_range:<20} {scale:.2f}")

    print("-" * 100)
    if len(scales) == 1:
        scale = scales.pop()
        print(f"\nConsistent scale factor detected: {scale}x")
        if abs(scale - 1.0) < 1e-6:
            print("=> LR and GT are the SAME resolution. "
                  "A standard Residual U-Net (no upsampling) is sufficient.")
        else:
            print(f"=> LR and GT differ by {scale}x. "
                  f"You need a Super-Resolution U-Net with {scale}x upsampling.")
    else:
        print(f"\nWARNING: inconsistent scale factors found across samples: {scales}")
        print("Your dataset may mix 256->512 and 128->256 pairs. "
              "Handle this by resizing/padding in the Dataset class, "
              "or by training separate models per scale.")

    # Check whether GT and LR dtypes match (affects normalization code)
    print(f"\nGT dtype: {gt.dtype}, LR dtype: {lr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to train/ directory containing GT/ and NoisyLR/")
    parser.add_argument("--n_samples", type=int, default=20)
    args = parser.parse_args()

    inspect(args.root, args.n_samples)
