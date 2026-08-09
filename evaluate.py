"""
evaluate.py

Standalone evaluation of a trained checkpoint on:
  - the existing in-distribution (ID) validation set (organizer split,
    same val_fraction/seed as training -> same stems every time), and
  - deterministic out-of-distribution (OOD) variants built by applying
    fixed extra perturbations on top of the *existing* validation NoisyLR
    inputs (see ood.py). GT is never modified and is always the original
    organizer-provided ground truth.

Reports PSNR, SSIM, and LPIPS for ID and for each OOD variant, and writes
everything to a JSON results file for reproducible, checkpointed comparison
across experiments.

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
  - LPIPS is optional at import/runtime. If the `lpips` package or its
    pretrained weights are unavailable, LPIPS is skipped (reported as
    null in the JSON) and a warning is printed; PSNR/SSIM are unaffected.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    Image = None
    _HAS_PIL = False

from dataset import IMAGE_EXTENSIONS, _load_array, make_train_val_split
from metrics import compute_psnr, compute_ssim
from models import build_model
from ood import apply_ood_variant, variant_names
from ood_permutations import (
    ALL_PERMUTATIONS,
    apply_permutation,
    order_key,
    permutation_labels,
)
from utils import load_config

# -----------------------------------------------------------------------------
# Tweakable composite-score constants
# -----------------------------------------------------------------------------
COMPOSITE_WEIGHTS = {
    "psnr": 0.40,
    "ssim": 0.40,
    "lpips": 0.20,
}
COMPOSITE_PSNR_REF_DB = 40.0  # PSNR normalization reference
COMPOSITE_LPIPS_REF = 0.50  # LPIPS normalization reference
COMPOSITE_SECTION_WEIGHT_ID = 1.0
COMPOSITE_SECTION_WEIGHT_OOD = 1.0
# Order-generalization contributes its *worst-case* permutation to the
# composite score (not the mean) -- the whole point of this section is to
# penalize models that look good on average but fail badly on some order.
COMPOSITE_SECTION_WEIGHT_ORDER_WORST_CASE = 1.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_safe(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable Python objects.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))

    return str(obj)


def _scale_pair_key(lr: torch.Tensor, gt: torch.Tensor) -> str:
    """
    Derive the scale-pair label from tensor shapes.
    Examples:
        GT 512x512, LR 256x256 -> "512_to_256"
        GT 256x256, LR 128x128 -> "256_to_128"
    """
    gt_h, gt_w = int(gt.shape[-2]), int(gt.shape[-1])
    lr_h, lr_w = int(lr.shape[-2]), int(lr.shape[-1])

    if gt_h == gt_w and lr_h == lr_w and gt_h == lr_h * 2 and gt_w == lr_w * 2:
        return f"{gt_h}_to_{lr_h}"

    if gt_h == lr_h * 2 and gt_w == lr_w * 2:
        return f"{gt_h}x{gt_w}_to_{lr_h}x{lr_w}"

    return f"{gt_h}x{gt_w}_to_{lr_h}x{lr_w}"


def _empty_metric_bucket() -> Dict[str, float]:
    return {
        "psnr_sum": 0.0,
        "ssim_sum": 0.0,
        "lpips_sum": 0.0,
        "lpips_count": 0.0,
        "n_samples": 0.0,
    }


def _update_metric_bucket(bucket: Dict[str, float], psnr: float, ssim: float, lpips: Optional[float]) -> None:
    bucket["psnr_sum"] += float(psnr)
    bucket["ssim_sum"] += float(ssim)
    if lpips is not None:
        bucket["lpips_sum"] += float(lpips)
        bucket["lpips_count"] += 1.0
    bucket["n_samples"] += 1.0


def _finalize_metric_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    n = int(bucket["n_samples"])
    lpips = None
    if int(bucket["lpips_count"]) > 0:
        lpips = bucket["lpips_sum"] / bucket["lpips_count"]

    return {
        "psnr": bucket["psnr_sum"] / max(n, 1),
        "ssim": bucket["ssim_sum"] / max(n, 1),
        "lpips": lpips,
        "n_samples": n,
    }


def _finalize_nested_scale_buckets(scale_buckets: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {scale_key: _finalize_metric_bucket(bucket) for scale_key, bucket in sorted(scale_buckets.items())}


def _compute_single_sample_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lpips_metric=None,
) -> Tuple[float, float, Optional[float]]:
    """
    pred, gt: [1, 1, H, W]
    """
    data_range = gt.amax(dim=[1, 2, 3]) - gt.amin(dim=[1, 2, 3])
    data_range = torch.clamp(data_range, min=1e-6)

    psnr = compute_psnr(pred, gt, data_range).item()
    ssim = compute_ssim(pred, gt, data_range).item()
    lpips = lpips_metric(pred, gt) if lpips_metric is not None else None
    return psnr, ssim, lpips


def _resolve_lpips_enabled(cfg: dict, cli_value: Optional[bool]) -> bool:
    """
    Default intent: enabled. Explicit CLI overrides win, then config value,
    then True if nothing is specified.
    """
    if cli_value is not None:
        return bool(cli_value)

    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
    if "compute_lpips" in metrics_cfg:
        return bool(metrics_cfg.get("compute_lpips"))

    return True


def _resolve_lpips_net(cfg: dict, cli_lpips_net: Optional[str] = None) -> str:
    if cli_lpips_net:
        return cli_lpips_net
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg, dict) else {}
    return metrics_cfg.get("lpips_net", "alex")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    cfg_override: Optional[dict] = None,
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = cfg_override if cfg_override is not None else ckpt.get("cfg")
    if cfg is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no embedded 'cfg' and no --config was provided.")

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
def evaluate_id(
    model,
    val_ds,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    num_workers: int,
    lpips_metric=None,
) -> Dict[str, Any]:
    """
    Standard ID evaluation: model(NoisyLR) vs GT, unmodified, in the
    existing validation split's natural iteration order (no shuffling).

    Returns an overall aggregate plus a by_scale breakdown.
    """
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    overall = _empty_metric_bucket()
    by_scale: Dict[str, Dict[str, float]] = {}

    for batch in loader:
        lr_batch = batch["lr"]
        gt_batch = batch["gt"].to(device, non_blocking=True)
        pred_batch = _forward(model, lr_batch, device, use_amp)

        bsz = gt_batch.size(0)
        for i in range(bsz):
            lr_i = lr_batch[i : i + 1]
            gt_i = gt_batch[i : i + 1]
            pred_i = pred_batch[i : i + 1]
            psnr, ssim, lpips = _compute_single_sample_metrics(pred_i, gt_i, lpips_metric=lpips_metric)
            scale_key = _scale_pair_key(lr_i, gt_i)

            _update_metric_bucket(overall, psnr, ssim, lpips)
            if scale_key not in by_scale:
                by_scale[scale_key] = _empty_metric_bucket()
            _update_metric_bucket(by_scale[scale_key], psnr, ssim, lpips)

    return {
        "overall": _finalize_metric_bucket(overall),
        "by_scale": _finalize_nested_scale_buckets(by_scale),
    }


@torch.no_grad()
def evaluate_ood(
    model,
    val_ds,
    variant: str,
    device: torch.device,
    use_amp: bool,
    lpips_metric=None,
) -> Dict[str, Any]:
    """
    OOD evaluation for a single variant: for each validation sample,
    deterministically perturbs the *existing* NoisyLR (see ood.py), runs
    the model, and compares against the original, untouched GT.

    Iterates one sample at a time (no batching) so that `sample_index`
    passed to apply_ood_variant is simply the position in the fixed val
    stem order -- keeping the seed derivation in ood.py trivial to reason
    about and independent of batch size / world size.

    Returns an overall aggregate plus a by_scale breakdown.
    """
    overall = _empty_metric_bucket()
    by_scale: Dict[str, Dict[str, float]] = {}

    for idx in range(len(val_ds)):
        sample = val_ds[idx]
        lr = sample["lr"]  # [1, H, W], original NoisyLR
        gt = sample["gt"].unsqueeze(0).to(device)  # [1, 1, H, W], untouched GT

        lr_ood = apply_ood_variant(lr, variant, sample_index=idx).unsqueeze(0)  # [1, 1, H, W]
        pred = _forward(model, lr_ood, device, use_amp)

        psnr, ssim, lpips = _compute_single_sample_metrics(pred, gt, lpips_metric=lpips_metric)
        scale_key = _scale_pair_key(sample["lr"].unsqueeze(0), gt)

        _update_metric_bucket(overall, psnr, ssim, lpips)
        if scale_key not in by_scale:
            by_scale[scale_key] = _empty_metric_bucket()
        _update_metric_bucket(by_scale[scale_key], psnr, ssim, lpips)

    return {
        "overall": _finalize_metric_bucket(overall),
        "by_scale": _finalize_nested_scale_buckets(by_scale),
    }


@torch.no_grad()
def evaluate_permutation(
    model,
    val_ds,
    order: Tuple[str, str, str],
    scale: int,
    device: torch.device,
    use_amp: bool,
    lpips_metric=None,
) -> Dict[str, Any]:
    """
    Order-generalization evaluation for one specific S/G/D ordering.

    Unlike evaluate_ood (which perturbs the existing, already-downsampled
    NoisyLR), this rebuilds the LR input from the *clean GT* by applying
    speckle noise (S), additive Gaussian noise (G), and downsampling (D)
    in the given order -- so all 6 permutations are genuinely distinct,
    including which point downsampling happens at.

    Returns an overall aggregate plus a by_scale breakdown, same shape as
    evaluate_id / evaluate_ood so downstream code (composite score, JSON
    writers) can treat it uniformly.
    """
    overall = _empty_metric_bucket()
    by_scale: Dict[str, Dict[str, float]] = {}

    for idx in range(len(val_ds)):
        sample = val_ds[idx]
        gt = sample["gt"].unsqueeze(0).to(device)  # [1, 1, H, W], untouched GT

        gt_cpu = sample["gt"]  # [1, H, W], native scale, on CPU (matches ood.py convention)
        lr_perm = apply_permutation(gt_cpu, order, scale=scale, sample_index=idx).unsqueeze(0)  # [1,1,h,w]

        pred = _forward(model, lr_perm, device, use_amp)

        psnr, ssim, lpips = _compute_single_sample_metrics(pred, gt, lpips_metric=lpips_metric)
        scale_key = _scale_pair_key(lr_perm, gt)

        _update_metric_bucket(overall, psnr, ssim, lpips)
        if scale_key not in by_scale:
            by_scale[scale_key] = _empty_metric_bucket()
        _update_metric_bucket(by_scale[scale_key], psnr, ssim, lpips)

    return {
        "overall": _finalize_metric_bucket(overall),
        "by_scale": _finalize_nested_scale_buckets(by_scale),
    }


def evaluate_order_generalization(
    model,
    val_ds,
    scale: int,
    device: torch.device,
    use_amp: bool,
    lpips_metric=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs evaluate_permutation for all 6 S/G/D orderings and computes the
    order-generalization summary the organizers' clue calls for:

        1. mean       -- average(P1...P6), overall robustness
        2. worst_case -- min(P1...P6), how badly the model can fail when
                          the order changes (the critical one, since the
                          hidden test order is unknown)
        3. order_std  -- population std across the 6 orders, how sensitive
                          the model is to degradation order (lower is
                          better)

    Computed separately for PSNR, SSIM, and LPIPS (when available).

    Result shape (matches the "one complete JSON" layout so exp1/exp3/exp4
    runs are directly diffable):
        {
            "S_G_D": {"overall": {...}, "by_scale": {...}},
            "S_D_G": {...}, "G_S_D": {...}, "G_D_S": {...},
            "D_S_G": {...}, "D_G_S": {...},
            "mean":       {"psnr": .., "ssim": .., "lpips": ..},
            "worst_case": {"psnr": .., "ssim": .., "lpips": ..},
            "order_std":  {"psnr": .., "ssim": .., "lpips": ..},
        }
    """
    order_gen: Dict[str, Any] = {}
    order_keys: List[str] = []

    for spec in ALL_PERMUTATIONS:
        ok = order_key(spec.order)  # e.g. "S->G->D"
        json_key = ok.replace("->", "_")  # e.g. "S_G_D", matches requested schema
        order_keys.append(json_key)

        if verbose:
            print(f"Evaluating permutation {spec.label} ({ok}) ({len(val_ds)} samples)...")
        r = evaluate_permutation(model, val_ds, spec.order, scale, device, use_amp, lpips_metric=lpips_metric)
        order_gen[json_key] = r
        o = r["overall"]
        if verbose:
            print(f"  {spec.label} ({ok}): PSNR={o['psnr']:.3f}dB SSIM={o['ssim']:.4f} LPIPS={o['lpips']}")

    def _stats_for_metric(metric_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        values = [order_gen[k]["overall"][metric_name] for k in order_keys]
        values = [v for v in values if v is not None]
        if not values:
            return None, None, None

        mean = float(sum(values) / len(values))
        if metric_name == "lpips":
            worst_case = float(max(values))
        else:
            worst_case = float(min(values))
        variance = sum((v - mean) ** 2 for v in values) / len(values)  # population std
        order_std = float(variance**0.5)
        return mean, worst_case, order_std

    mean_out: Dict[str, Optional[float]] = {}
    worst_out: Dict[str, Optional[float]] = {}
    std_out: Dict[str, Optional[float]] = {}

    for metric_name in ("psnr", "ssim", "lpips"):
        mean, worst_case, order_std = _stats_for_metric(metric_name)
        mean_out[metric_name] = mean
        worst_out[metric_name] = worst_case
        std_out[metric_name] = order_std

    order_gen["mean"] = mean_out
    order_gen["worst_case"] = worst_out
    order_gen["order_std"] = std_out

    if verbose and mean_out["psnr"] is not None:
        print(
            f"  Summary: PSNR mean={mean_out['psnr']:.3f}dB "
            f"worst_case={worst_out['psnr']:.3f}dB "
            f"order_std={std_out['psnr']:.3f}dB"
        )

    return order_gen


def _normalize_metric_component(value: Optional[float], metric_name: str) -> float:
    if value is None:
        return 0.0

    if metric_name == "psnr":
        return float(np.clip(value / COMPOSITE_PSNR_REF_DB, 0.0, 1.0))

    if metric_name == "ssim":
        return float(np.clip(value, 0.0, 1.0))

    if metric_name == "lpips":
        return float(np.clip(1.0 - (value / COMPOSITE_LPIPS_REF), 0.0, 1.0))

    raise ValueError(f"Unknown metric name: {metric_name}")


def _section_score(section: Dict[str, Any]) -> float:
    weights = COMPOSITE_WEIGHTS.copy()
    components = []
    active_weight_sum = 0.0

    psnr_score = _normalize_metric_component(section.get("psnr"), "psnr")
    ssim_score = _normalize_metric_component(section.get("ssim"), "ssim")
    components.append(weights["psnr"] * psnr_score)
    active_weight_sum += weights["psnr"]

    components.append(weights["ssim"] * ssim_score)
    active_weight_sum += weights["ssim"]

    if section.get("lpips") is not None:
        lpips_score = _normalize_metric_component(section.get("lpips"), "lpips")
        components.append(weights["lpips"] * lpips_score)
        active_weight_sum += weights["lpips"]

    if active_weight_sum <= 0:
        return 0.0

    return float(sum(components) / active_weight_sum)


def _compute_composite_score(results: Dict[str, Any]) -> float:
    """
    A quick-to-scan scalar summary over ID, all OOD variants, and (when
    present) the order-generalization worst-case.

    Uses the overall ID score, the overall score of each OOD variant, and
    the worst-case-permutation score from order_generalization. The
    worst-case (not the mean-across-orders) is what feeds the composite,
    since a model that's great on average but collapses on one hidden-test
    order is exactly the failure mode this section exists to catch.
    """
    sections = []

    if "id" in results and "overall" in results["id"]:
        sections.append((_section_score(results["id"]["overall"]), COMPOSITE_SECTION_WEIGHT_ID))

    for variant_name, variant_result in results.get("ood", {}).items():
        if "overall" in variant_result:
            sections.append((_section_score(variant_result["overall"]), COMPOSITE_SECTION_WEIGHT_OOD))

    order_gen = results.get("order_generalization")
    if isinstance(order_gen, dict) and "worst_case" in order_gen:
        # worst_case is {"psnr": .., "ssim": .., "lpips": ..} -- reuse
        # _section_score's normalize-and-weight logic directly on it.
        wc = order_gen["worst_case"]
        if wc.get("psnr") is not None and wc.get("ssim") is not None:
            sections.append((_section_score(wc), COMPOSITE_SECTION_WEIGHT_ORDER_WORST_CASE))

    if not sections:
        return 0.0

    num = sum(score * weight for score, weight in sections)
    den = sum(weight for _, weight in sections)
    return float(num / max(den, 1e-12))


def _collect_benchmark_files(image_dir: str) -> List[str]:
    supported_exts = tuple(ext.lower() for ext in (IMAGE_EXTENSIONS + (".npy",)))
    files = [
        f
        for f in sorted(os.listdir(image_dir))
        if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() in supported_exts
    ]
    if not files:
        raise RuntimeError(f"No supported benchmark images found in {image_dir}. Expected extensions: {supported_exts}")
    return files


def _load_benchmark_tensor(path: str, device: torch.device) -> torch.Tensor:
    arr = _load_array(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D array/image at {path}, got shape {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float().to(device)


def _save_prediction(path: str, pred: torch.Tensor) -> None:
    pred_np = pred.squeeze().detach().cpu().numpy()

    if path.lower().endswith(".npy"):
        np.save(path, pred_np)
        return

    if not _HAS_PIL:
        raise RuntimeError(
            "Pillow is required to write benchmark image outputs. Install Pillow or use .npy benchmark inputs/outputs."
        )

    pred_np = np.clip(pred_np, 0.0, 1.0)
    Image.fromarray((pred_np * 255.0).astype(np.uint8)).save(path)


@torch.no_grad()
def _process_image_single(
    model, img_tensor: torch.Tensor, tile_size: Optional[int], overlap: int, scale: int
) -> torch.Tensor:
    """
    Single-image inference with optional tile fallback for very large images.
    """
    _, _, h, w = img_tensor.shape
    if tile_size is None or (h <= tile_size and w <= tile_size):
        return model(img_tensor)

    if tile_size <= overlap:
        raise ValueError("tile_overlap must be smaller than tile_size")

    out_h, out_w = h * scale, w * scale
    out_tensor = torch.zeros((1, 1, out_h, out_w), device=img_tensor.device, dtype=img_tensor.dtype)
    count_tensor = torch.zeros_like(out_tensor)

    stride = tile_size - overlap
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            t = min(i + tile_size, h)
            r = min(j + tile_size, w)
            b = i if (t - i) == tile_size else max(0, t - tile_size)
            l = j if (r - j) == tile_size else max(0, r - tile_size)

            tile = img_tensor[:, :, b:t, l:r]
            pred_tile = model(tile)

            out_tensor[:, :, b * scale : t * scale, l * scale : r * scale] += pred_tile
            count_tensor[:, :, b * scale : t * scale, l * scale : r * scale] += 1

    return out_tensor / count_tensor.clamp_min(1)


@torch.no_grad()
def benchmark_inference(
    model,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    tile_size: Optional[int],
    overlap: int,
    scale: int,
) -> Dict[str, Any]:
    """
    Benchmark real-image inference on files from disk.

    Starts a timer, loads images from `image_dir`, runs batched inference
    when shapes allow, writes outputs to `output_dir`, and returns a timing
    summary. Input/output formats are preserved per file extension.

    Returns:
        {
            total_seconds,
            n_images,
            ms_per_image_mean,
            io_read_seconds,
            inference_seconds,
            io_write_seconds,
        }
    """
    files = _collect_benchmark_files(image_dir)
    os.makedirs(output_dir, exist_ok=True)

    io_read_seconds = 0.0
    inference_seconds = 0.0
    io_write_seconds = 0.0

    t_total = time.perf_counter()

    batch_tensors: List[torch.Tensor] = []
    batch_fnames: List[str] = []
    batch_shapes: List[Tuple[int, int]] = []
    n_images = 0

    def flush_batch() -> None:
        nonlocal inference_seconds, io_write_seconds, n_images
        nonlocal batch_tensors, batch_fnames, batch_shapes

        if not batch_tensors:
            return

        same_shape = len(set(batch_shapes)) == 1
        use_batched_forward = same_shape and (
            tile_size is None or (batch_shapes[0][0] <= tile_size and batch_shapes[0][1] <= tile_size)
        )

        if use_batched_forward:
            batch = torch.cat(batch_tensors, dim=0)
            t_inf0 = time.perf_counter()
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type=device.type):
                    preds = model(batch)
            else:
                preds = model(batch)
            inference_seconds += time.perf_counter() - t_inf0

            t_write0 = time.perf_counter()
            for fname, pred in zip(batch_fnames, preds):
                _save_prediction(os.path.join(output_dir, fname), pred.unsqueeze(0))
                n_images += 1
            io_write_seconds += time.perf_counter() - t_write0
        else:
            for fname, img in zip(batch_fnames, batch_tensors):
                t_inf0 = time.perf_counter()
                if use_amp and device.type == "cuda":
                    with torch.autocast(device_type=device.type):
                        pred = _process_image_single(model, img, tile_size, overlap, scale)
                else:
                    pred = _process_image_single(model, img, tile_size, overlap, scale)
                inference_seconds += time.perf_counter() - t_inf0

                t_write0 = time.perf_counter()
                _save_prediction(os.path.join(output_dir, fname), pred)
                io_write_seconds += time.perf_counter() - t_write0
                n_images += 1

        batch_tensors = []
        batch_fnames = []
        batch_shapes = []

    for fname in files:
        in_path = os.path.join(image_dir, fname)

        t_read0 = time.perf_counter()
        img_t = _load_benchmark_tensor(in_path, device)
        io_read_seconds += time.perf_counter() - t_read0

        batch_tensors.append(img_t)
        batch_fnames.append(fname)
        batch_shapes.append(tuple(img_t.shape[-2:]))

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    total_seconds = time.perf_counter() - t_total
    ms_per_image_mean = (total_seconds / max(n_images, 1)) * 1000.0

    return {
        "total_seconds": total_seconds,
        "n_images": n_images,
        "ms_per_image_mean": ms_per_image_mean,
        "io_read_seconds": io_read_seconds,
        "inference_seconds": inference_seconds,
        "io_write_seconds": io_write_seconds,
        "input_dir": image_dir,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "tile_size": tile_size,
        "tile_overlap": overlap,
        "scale": scale,
    }


def run_full_evaluation(
    checkpoint_path: str,
    config_path: Optional[str],
    device: torch.device,
    lpips_net: Optional[str] = None,
    ood_variants: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    data_root: Optional[str] = None,
    compute_lpips: Optional[bool] = None,
    benchmark_dir: Optional[str] = None,
    benchmark_out: Optional[str] = None,
    run_order_generalization: bool = True,
) -> dict:
    cfg_override = load_config(config_path) if config_path else None
    model, cfg, epoch = load_model_from_checkpoint(checkpoint_path, device, cfg_override)

    cfg = copy.deepcopy(cfg)

    # Same override mechanism as train_ddp.py's --root.
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
    tile_size = cfg.get("inference", {}).get("tile_size", None)
    overlap = cfg.get("inference", {}).get("tile_overlap", 32)
    scale = int(cfg["model"].get("scale", 2))

    from lpips_metric import try_build_lpips

    lpips_enabled = _resolve_lpips_enabled(cfg, compute_lpips)
    lpips_net = _resolve_lpips_net(cfg, lpips_net)

    lpips_metric = None
    if lpips_enabled:
        lpips_metric = try_build_lpips(net=lpips_net, device=device)

    if ood_variants is None:
        ood_variants = variant_names()

    ckpt_size_mb = os.path.getsize(checkpoint_path) / 1e6
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "checkpoint": checkpoint_path,
        "config": _json_safe(cfg),
        "epoch": epoch,
        "n_val_samples": len(val_ds),
        "lpips_enabled": lpips_metric is not None,
        "lpips_net": lpips_net if lpips_metric is not None else None,
        "model": {
            "name": cfg["model"].get("name"),
            "class": model.__class__.__name__,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "checkpoint_size_mb": float(ckpt_size_mb),
        },
    }

    t0 = time.perf_counter()
    print(f"Evaluating ID validation set ({len(val_ds)} samples)...")
    id_result = evaluate_id(
        model,
        val_ds,
        device,
        use_amp,
        batch_size=batch_size,
        num_workers=num_workers,
        lpips_metric=lpips_metric,
    )
    results["id"] = id_result
    id_overall = id_result["overall"]
    print(f"  ID: PSNR={id_overall['psnr']:.3f}dB SSIM={id_overall['ssim']:.4f} LPIPS={id_overall['lpips']}")

    results["ood"] = {}
    for variant in ood_variants:
        print(f"Evaluating OOD variant '{variant}' ({len(val_ds)} samples)...")
        r = evaluate_ood(model, val_ds, variant, device, use_amp, lpips_metric=lpips_metric)
        # Strip the "ood_" prefix for the JSON key (e.g. "ood_gaussian" -> "gaussian")
        # so the combined results file reads as id / ood.gaussian / ood.speckle / ...
        json_key = variant[4:] if variant.startswith("ood_") else variant
        results["ood"][json_key] = r
        o = r["overall"]
        print(f"  {variant}: PSNR={o['psnr']:.3f}dB SSIM={o['ssim']:.4f} LPIPS={o['lpips']}")

    # Order generalization runs by default now, as part of the single
    # combined evaluation JSON (id / ood / order_generalization /
    # inference_benchmark all in one file, so exp1/exp3/exp4 runs are
    # directly diffable). Use --skip-order-generalization to opt out
    # (e.g. for a quick smoke-test run) since it's 6x the forward passes
    # of a single OOD variant.
    if run_order_generalization:
        print(f"Evaluating order generalization: 6 S/G/D permutations ({len(val_ds)} samples each)...")
        results["order_generalization"] = evaluate_order_generalization(
            model,
            val_ds,
            scale=scale,
            device=device,
            use_amp=use_amp,
            lpips_metric=lpips_metric,
        )

    results["metrics_eval_seconds"] = time.perf_counter() - t0

    if benchmark_dir is not None:
        if benchmark_out is None:
            benchmark_out = os.path.join(benchmark_dir, "restored_outputs")
        print(f"Running inference benchmark on {benchmark_dir} -> {benchmark_out} ...")
        results["inference_benchmark"] = benchmark_inference(
            model=model,
            image_dir=benchmark_dir,
            output_dir=benchmark_out,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
            tile_size=tile_size,
            overlap=overlap,
            scale=scale,
        )

    results["composite_score"] = _compute_composite_score(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on ID + deterministic OOD validation sets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint (e.g. best.pt).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML to use for dataset/model construction. Defaults to the cfg embedded in the checkpoint.",
    )
    parser.add_argument("--out", type=str, default=None, help="Where to write the JSON results file.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Override data.root from the config / checkpoint's embedded cfg (e.g. the actual "
        "dataset path on this machine, such as /kaggle/input/<dataset>/train).",
    )
    parser.add_argument(
        "--ood-variants",
        type=str,
        nargs="*",
        default=None,
        help="Subset of OOD variants to run (default: all of ood.variant_names()).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    lpips_group = parser.add_mutually_exclusive_group()
    lpips_group.add_argument("--lpips", dest="lpips", action="store_true", help="Force LPIPS evaluation on.")
    lpips_group.add_argument("--no-lpips", dest="lpips", action="store_false", help="Disable LPIPS evaluation.")
    parser.set_defaults(lpips=None)

    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Optional directory of raw images for a separate inference benchmark.",
    )
    parser.add_argument(
        "--benchmark-out",
        type=str,
        default=None,
        help="Optional output directory for restored benchmark images (defaults to <benchmark-dir>/restored_outputs).",
    )
    parser.add_argument(
        "--skip-order-generalization",
        action="store_true",
        help="Skip the 6-permutation S/G/D order-generalization evaluation (see "
        "ood_permutations.py). It runs by default as part of the combined results JSON "
        "(id / ood / order_generalization / inference_benchmark), since that's the "
        "organizer-mandated order-robustness check needed to compare Exp 1 / Exp 3 / Exp 4 "
        "on equal footing. Use this flag for a quick smoke-test run only -- it's 6x more "
        "forward passes than a single OOD variant.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    results = run_full_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        lpips_net=None,
        ood_variants=args.ood_variants,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.root,
        compute_lpips=args.lpips,
        benchmark_dir=args.benchmark_dir,
        benchmark_out=args.benchmark_out,
        run_order_generalization=not args.skip_order_generalization,
    )

    out_path = args.out
    if out_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint) or "."
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(ckpt_dir, f"eval_{ckpt_name}.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
