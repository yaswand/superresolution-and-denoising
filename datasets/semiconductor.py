"""Paired KLA .npy dataset.  Inputs intentionally remain unclipped."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _seeded_fraction(name: str) -> float:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class SemiconductorPairs(Dataset):
    def __init__(self, root: str | Path, cfg: dict[str, Any], train: bool, scale: int = 2):
        self.root, self.cfg, self.train, self.scale = Path(root), cfg, train, scale
        gt, noisy = self.root / "GT", self.root / "NoisyLR"
        names = sorted(p.stem for p in gt.glob("*.npy") if (noisy / f"{p.stem}.npy").exists())
        if not names: raise FileNotFoundError(f"No paired .npy files found under {self.root}")
        # Use the leading filename component as a structure group when available.
        group = lambda n: n.split("_")[0] if cfg.get("split_mode") == "prefix_group" else n
        val_fraction = float(cfg.get("val_fraction", .15))
        self.names = [n for n in names if (_seeded_fraction(group(n)) >= val_fraction) == train]
        self.gt, self.noisy = gt, noisy

    def __len__(self): return len(self.names)

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int], mode="bicubic"):
        kwargs = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
        return F.interpolate(x[None], size=size, mode=mode, **kwargs)[0]

    def _augment(self, gt: torch.Tensor, supplied: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        # Independent choices deliberately create unseen noise/downsample combinations.
        source = gt if torch.rand(()) < .5 else supplied
        if source.shape[-2:] != gt.shape[-2:]:
            source = self._resize(source, gt.shape[-2:])
        scale = 4 if torch.rand(()) < float(c.get("scale4_probability", .15)) else self.scale
        h, w = gt.shape[-2:]
        low = self._resize(source, (max(1, h // scale), max(1, w // scale)), "area")
        low = self._resize(low, (h // self.scale, w // self.scale)) if scale != self.scale else low
        speckle = torch.empty(()).uniform_(*c.get("speckle_sigma", [0., .12]))
        gaussian = torch.empty(()).uniform_(*c.get("gaussian_sigma", [0., .04]))
        return low + torch.randn_like(low) * (speckle * low + gaussian)

    def _crop_and_orient(self, low: torch.Tensor, gt: torch.Tensor):
        patch = int(self.cfg.get("patch_size", 128)); h, w = low.shape[-2:]
        patch = min(patch, h, w); y = torch.randint(0, h - patch + 1, ()).item(); x = torch.randint(0, w - patch + 1, ()).item()
        low = low[:, y:y+patch, x:x+patch]
        s = gt.shape[-1] // w
        gt = gt[:, y*s:(y+patch)*s, x*s:(x+patch)*s]
        k = torch.randint(0, 4, ()).item(); low, gt = torch.rot90(low, k, (-2,-1)), torch.rot90(gt, k, (-2,-1))
        if torch.rand(()) < .5: low, gt = low.flip(-1), gt.flip(-1)
        if torch.rand(()) < .5: low, gt = low.flip(-2), gt.flip(-2)
        return low, gt

    def __getitem__(self, i):
        name = self.names[i]
        gt = torch.from_numpy(np.load(self.gt / f"{name}.npy").astype(np.float32, copy=False))[None]
        low = torch.from_numpy(np.load(self.noisy / f"{name}.npy").astype(np.float32, copy=False))[None]
        # Native pairs are 2x; make a consistent low-resolution input for a 4x run.
        expected = (gt.shape[-2] // self.scale, gt.shape[-1] // self.scale)
        if low.shape[-2:] != expected:
            low = self._resize(low, expected, "area")
        if self.train and torch.rand(()) < float(self.cfg.get("synthetic_probability", .55)):
            low = self._augment(gt, low)
        if self.train: low, gt = self._crop_and_orient(low, gt)
        return {"input": low, "target": gt, "name": name, "ood": False}


def build_loaders(cfg: dict, scale: int):
    dcfg = cfg["data"]
    train = SemiconductorPairs(dcfg["root"], dcfg, True, scale)
    val = SemiconductorPairs(dcfg["root"], dcfg, False, scale)
    workers = int(dcfg.get("workers", 0))
    # On Windows, multiple workers can exceed the page-file-backed shared-memory
    # mapping limit.  workers=0 loads batches in the main process and avoids it.
    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    if workers > 0:
        kw["persistent_workers"] = True
    return (DataLoader(train, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True, **kw),
            DataLoader(val, batch_size=1, shuffle=False, **kw))
