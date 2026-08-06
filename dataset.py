"""
dataset.py

Dataset pipeline for the grayscale image restoration challenge
(2x super-resolution + denoising on paired GT / NoisyLR .npy arrays).

Normalization policy (IMPORTANT):
    The organizers' data note states that degraded LR pixel values may
    legitimately fall outside the GT's [0, 1] range (speckle noise pushes
    values beyond the original signal), and the model must handle this.

    Consequently this dataset does NOT apply any per-image normalization
    (no z-score, no min-max rescaling) and does NOT clip the LR input.
    Both GT and LR are loaded as float32 and fed to the model in their
    native numerical scale. GT is expected to be approximately in [0, 1];
    LR is close to that scale but may exceed it -- that excess is real
    degradation signal, not an artifact to normalize away.

    Downstream consequences:
      - No per-sample mean/std needs to be carried through the batch for
        de-normalization (there is nothing to de-normalize).
      - metrics.py compares prediction vs GT directly in this native scale,
        clamping the *prediction* only where a metric implementation
        requires a bounded range (e.g. SSIM's constants), never the input.
      - infer.py clamps the final prediction to [0, 1] only at export time,
        for writing a valid image file -- never before or during the
        forward pass.

Data layout:
    root/GT/xxxxxx.npy       -> ground truth, shape [H, W]
    root/NoisyLR/xxxxxx.npy  -> degraded input, shape [H/scale, W/scale]

    Also supports .png / .jpg / .jpeg pairs with the same stem, so a mixed
    or non-.npy dataset works without code changes (pixel values are read
    as float32; 8-bit image formats are scaled to [0, 1] on load since
    that is their native representable range, unlike .npy which is trusted
    as-is).

Augmentation:
    Paired (applied identically to LR and GT, keeping them pixel-aligned):
      - horizontal flip
      - vertical flip
      - random 90-degree rotation
      - aligned random crop (crop size given in LR pixels; GT crop is
        scale times larger, at the corresponding location)

    LR-only (applied after the paired crop, never touching GT, for
    robustness against out-of-distribution degradation at eval time):
      - mild Gaussian noise
      - mild speckle (multiplicative) noise
      - mild Gaussian blur
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _load_array(path: str) -> np.ndarray:
    """
    Loads a single-channel array from disk as float32, native scale.

    .npy files are trusted as-is (no rescaling) since the organizers'
    dataset is already stored as float32 on a consistent intensity scale.

    8-bit image formats (.png/.jpg/.jpeg) are divided by 255.0 on load,
    since that is the only sensible native scale for integer pixel data --
    this is a format-driven decode step, not a per-image normalization.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    elif ext in IMAGE_EXTENSIONS:
        if not _HAS_PIL:
            raise RuntimeError(f"Pillow is required to read {path}. Install with `pip install Pillow`.")
        with Image.open(path) as im:
            im = im.convert("L")  # grayscale
            arr = np.asarray(im, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported file extension for {path}: {ext}")

    if arr.ndim == 3:
        # Some arrays may be stored as [H, W, 1]; squeeze to [H, W].
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected single-channel array, got shape {arr.shape} at {path}")

    return arr


def _find_stem_file(directory: str, stem: str, extensions: Tuple[str, ...]) -> str:
    """Finds a file `stem.<ext>` in `directory` for the first matching extension."""
    for ext in extensions:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No file with stem '{stem}' and extensions {extensions} found in {directory}")


def list_paired_stems(root: str, extensions: Tuple[str, ...] = (".npy", ".png", ".jpg", ".jpeg")) -> List[str]:
    """
    Scans root/GT for files with any of `extensions`, returns sorted stems
    (filename without extension) that also have a matching file in
    root/NoisyLR. This makes the dataset robust to a mixed-format directory
    and to filename extension mismatches between GT and NoisyLR.
    """
    gt_dir = os.path.join(root, "GT")
    lr_dir = os.path.join(root, "NoisyLR")

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"NoisyLR directory not found: {lr_dir}")

    gt_files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in extensions]
    stems = sorted(os.path.splitext(f)[0] for f in gt_files)

    valid_stems = []
    for stem in stems:
        try:
            _find_stem_file(lr_dir, stem, extensions)
            valid_stems.append(stem)
        except FileNotFoundError:
            continue  # skip GT files with no matching LR pair

    if not valid_stems:
        raise RuntimeError(f"No paired GT/NoisyLR files found under {root}")

    return valid_stems


def _paired_random_crop(lr: np.ndarray, gt: np.ndarray, patch_size: int, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligned random crop: picks a patch_size x patch_size window in LR space,
    and crops the corresponding (patch_size*scale) x (patch_size*scale)
    window in GT space at the matching location.
    """
    lr_h, lr_w = lr.shape
    if lr_h < patch_size or lr_w < patch_size:
        raise ValueError(
            f"LR image ({lr_h}x{lr_w}) is smaller than patch_size ({patch_size}). Reduce data.patch_size in the config."
        )

    top = random.randint(0, lr_h - patch_size)
    left = random.randint(0, lr_w - patch_size)

    lr_crop = lr[top : top + patch_size, left : left + patch_size]
    gt_top, gt_left = top * scale, left * scale
    gt_patch = patch_size * scale
    gt_crop = gt[gt_top : gt_top + gt_patch, gt_left : gt_left + gt_patch]

    return lr_crop, gt_crop


def _paired_geometric_augment(lr: np.ndarray, gt: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Applies flips/rotation identically to LR and GT so they stay aligned."""
    if cfg.get("horizontal_flip", False) and random.random() < 0.5:
        lr = np.ascontiguousarray(lr[:, ::-1])
        gt = np.ascontiguousarray(gt[:, ::-1])

    if cfg.get("vertical_flip", False) and random.random() < 0.5:
        lr = np.ascontiguousarray(lr[::-1, :])
        gt = np.ascontiguousarray(gt[::-1, :])

    if cfg.get("random_rotate90", False):
        k = random.randint(0, 3)
        if k > 0:
            lr = np.ascontiguousarray(np.rot90(lr, k))
            gt = np.ascontiguousarray(np.rot90(gt, k))

    return lr, gt


def _apply_lr_only_degradation(lr: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Applies mild synthetic degradation to the LR tensor only, for robustness
    against out-of-distribution degraded inputs. Never touches GT. Operates
    additively/multiplicatively in the tensor's native scale -- no clamping,
    since out-of-range LR values are legitimate per the task brief.
    """
    noise_cfg = cfg.get("lr_gaussian_noise", {})
    if noise_cfg.get("enabled", False) and random.random() < noise_cfg.get("prob", 0.0):
        lo, hi = noise_cfg.get("sigma_range", [0.0, 0.02])
        sigma = random.uniform(lo, hi)
        lr = lr + torch.randn_like(lr) * sigma

    speckle_cfg = cfg.get("lr_speckle_noise", {})
    if speckle_cfg.get("enabled", False) and random.random() < speckle_cfg.get("prob", 0.0):
        lo, hi = speckle_cfg.get("sigma_range", [0.0, 0.03])
        sigma = random.uniform(lo, hi)
        lr = lr + lr * torch.randn_like(lr) * sigma

    blur_cfg = cfg.get("lr_gaussian_blur", {})
    if blur_cfg.get("enabled", False) and random.random() < blur_cfg.get("prob", 0.0):
        k = blur_cfg.get("kernel_size", 3)
        lo, hi = blur_cfg.get("sigma_range", [0.1, 0.8])
        sigma = random.uniform(lo, hi)
        lr = _gaussian_blur_tensor(lr, kernel_size=k, sigma=sigma)

    return lr


def _gaussian_blur_tensor(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    x: [1, H, W]. Applies a fixed Gaussian blur kernel via conv2d.
    Reflect padding avoids introducing border artifacts near LR-only noise.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    x = x.unsqueeze(0)  # [1, 1, H, W]
    pad = kernel_size // 2
    x = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    x = F.conv2d(x, kernel_2d)
    return x.squeeze(0)


class RestorationDataset(Dataset):
    """
    Loads (NoisyLR, GT) pairs from root/GT and root/NoisyLR.

    Returns:
        {
            "lr": FloatTensor [1, h, w]   (native scale, no normalization)
            "gt": FloatTensor [1, h*scale, w*scale]  (native scale, ~[0, 1])
            "fname": str  (stem, without extension)
        }

    Args:
        root: directory containing GT/ and NoisyLR/ subfolders.
        stems: list of filename stems to include (for train/val split).
        scale: LR -> GT upsampling factor.
        extensions: accepted file extensions, tried in order per stem.
        patch_size: aligned random crop size in LR pixels. None disables cropping.
        augment_cfg: augmentation sub-config (see configs/default.yaml). None disables all augmentation.
        train: if False, disables all augmentation regardless of augment_cfg
            (validation/test should see unaugmented, un-cropped-by-default data).
    """

    def __init__(
        self,
        root: str,
        stems: List[str],
        scale: int = 2,
        extensions: Tuple[str, ...] = (".npy", ".png", ".jpg", ".jpeg"),
        patch_size: Optional[int] = 96,
        augment_cfg: Optional[dict] = None,
        train: bool = True,
    ):
        self.gt_dir = os.path.join(root, "GT")
        self.lr_dir = os.path.join(root, "NoisyLR")
        self.stems = stems
        self.scale = scale
        self.extensions = extensions
        self.patch_size = patch_size if train else None
        self.augment_cfg = augment_cfg or {}
        self.train = train

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        gt_path = _find_stem_file(self.gt_dir, stem, self.extensions)
        lr_path = _find_stem_file(self.lr_dir, stem, self.extensions)

        gt = _load_array(gt_path)
        lr = _load_array(lr_path)

        expected_h, expected_w = lr.shape[0] * self.scale, lr.shape[1] * self.scale
        if gt.shape != (expected_h, expected_w):
            raise ValueError(
                f"Shape mismatch for stem '{stem}': GT {gt.shape} does not match "
                f"LR {lr.shape} * scale {self.scale} = ({expected_h}, {expected_w})"
            )

        if self.train and self.patch_size is not None:
            lr, gt = _paired_random_crop(lr, gt, self.patch_size, self.scale)

        if self.train:
            lr, gt = _paired_geometric_augment(lr, gt, self.augment_cfg)

        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0).float()

        if self.train:
            lr_tensor = _apply_lr_only_degradation(lr_tensor, self.augment_cfg)

        return {
            "lr": lr_tensor,
            "gt": gt_tensor,
            "fname": stem,
        }


def make_train_val_split(
    root: str,
    scale: int = 2,
    extensions: Tuple[str, ...] = (".npy", ".png", ".jpg", ".jpeg"),
    val_fraction: float = 0.1,
    patch_size: Optional[int] = 96,
    augment_cfg: Optional[dict] = None,
    seed: int = 42,
) -> Tuple[RestorationDataset, RestorationDataset]:
    """
    Splits paired stems at the filename level (before wrapping in Dataset
    objects) so train/val never overlap, using a seeded shuffle for
    reproducibility. Training set gets cropping + full augmentation;
    validation set sees full (uncropped) images with no augmentation.
    """
    all_stems = list_paired_stems(root, extensions)

    rng = random.Random(seed)
    shuffled = all_stems.copy()
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_fraction)
    val_stems = sorted(shuffled[:n_val])
    train_stems = sorted(shuffled[n_val:])

    train_ds = RestorationDataset(
        root,
        train_stems,
        scale=scale,
        extensions=extensions,
        patch_size=patch_size,
        augment_cfg=augment_cfg,
        train=True,
    )
    val_ds = RestorationDataset(
        root,
        val_stems,
        scale=scale,
        extensions=extensions,
        patch_size=None,
        augment_cfg=None,
        train=False,
    )
    return train_ds, val_ds
