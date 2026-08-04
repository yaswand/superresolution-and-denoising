"""
dataset.py

Dataset class for the KLA semiconductor image restoration challenge.

Key design decisions (from the project brief):
  - No handcrafted denoising before the network -> only normalization allowed.
  - Speckle noise can push NoisyLR values outside GT's range -> never clip.
  - LR and GT normalization stats are computed SEPARATELY (per image, per tensor)
    because they can have genuinely different distributions (unclipped speckle
    noise inflates NoisyLR's range beyond GT's). Using GT's mean/std on the LR
    image (or vice versa) would bias the residual the network has to learn.
  - We keep the un-normalized GT mean/std around so we can de-normalize model
    output back to pixel space for PSNR/SSIM computation.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def normalize(img: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """Per-image z-score normalization. Returns (normalized, mean, std)."""
    img = img.astype(np.float32)
    mean = float(img.mean())
    std = float(img.std())
    norm = (img - mean) / (std + eps)
    return norm, mean, std


class SemiconductorSRDataset(Dataset):
    """
    Loads (NoisyLR, GT) pairs from:
        root/GT/xxxxxx.npy
        root/NoisyLR/xxxxxx.npy

    Returns a dict so the training loop can access de-normalization stats
    without re-reading files:
        {
            "lr": FloatTensor [1, H, W]   (normalized)
            "gt": FloatTensor [1, 2H, 2W] (normalized)
            "gt_mean": float
            "gt_std": float
            "fname": str
        }
    """

    def __init__(self, root: str, filenames: Optional[list] = None):
        self.gt_dir = os.path.join(root, "GT")
        self.lr_dir = os.path.join(root, "NoisyLR")

        if filenames is not None:
            self.filenames = filenames
        else:
            self.filenames = sorted(os.listdir(self.gt_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]

        gt = np.load(os.path.join(self.gt_dir, fname))
        lr = np.load(os.path.join(self.lr_dir, fname))

        # Normalize independently -- see module docstring for why.
        lr_norm, lr_mean, lr_std = normalize(lr)
        gt_norm, gt_mean, gt_std = normalize(gt)

        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).float()

        return {
            "lr": lr_tensor,
            "gt": gt_tensor,
            "gt_mean": gt_mean,
            "gt_std": gt_std,
            "lr_mean": lr_mean,
            "lr_std": lr_std,
            "fname": fname,
        }


def make_train_val_split(
    root: str, val_fraction: float = 0.1, seed: int = 42
) -> Tuple[SemiconductorSRDataset, SemiconductorSRDataset]:
    """
    Splits the dataset at the FILENAME level (before wrapping in Dataset objects)
    so train/val never overlap, and returns two independent Dataset instances.
    """
    all_files = sorted(os.listdir(os.path.join(root, "GT")))
    n_val = int(len(all_files) * val_fraction)
    n_train = len(all_files) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(len(all_files)), [n_train, n_val], generator=generator
    )

    train_files = [all_files[i] for i in train_idx.indices]
    val_files = [all_files[i] for i in val_idx.indices]

    train_ds = SemiconductorSRDataset(root, filenames=train_files)
    val_ds = SemiconductorSRDataset(root, filenames=val_files)
    return train_ds, val_ds
