"""
ood.py

Deterministic out-of-distribution (OOD) validation variants.

Per the task brief, we do NOT invent a new GT -> LR degradation pipeline.
Instead we keep the organizer-provided paired validation data structure
fixed (GT stays exactly as-is) and apply deterministic extra perturbations
on top of the *existing* validation NoisyLR input to build harder inputs.
The model prediction is still compared against the original, untouched GT.

This preserves the real task structure: "restore this degraded input to
match this GT", just with an extra, fixed, known perturbation stacked on
top of whatever degradation the organizers already baked into NoisyLR.

Determinism:
    Every OOD variant is generated with a fixed seed that depends only on
    (variant name, sample index) -- never on wall-clock time, torch's
    global RNG state, or DataLoader worker id. This guarantees that the
    exact same OOD sample is produced every time a given checkpoint (or a
    different checkpoint) is evaluated, so PSNR/SSIM/LPIPS comparisons
    across checkpoints/experiments are apples-to-apples.

Variants:
    ood_gaussian - additive Gaussian noise on the NoisyLR tensor.
    ood_speckle  - multiplicative (speckle) noise on the NoisyLR tensor.
    ood_blur     - Gaussian blur on the NoisyLR tensor.
    ood_mixed    - blur, then additive Gaussian, then speckle, in sequence.

These deliberately reuse the same *style* of perturbation as the training
augmentation, but:
  - use fixed (not random-per-epoch) seeds and fixed severity, so the OOD
    benchmark is a static, reusable eval set rather than a moving target,
  - are typically applied at moderate-to-higher severity than the training
    augmentation ranges, so they actually probe generalization rather than
    just replaying what the model has already seen in-distribution.

Never touches GT. Operates directly in the LR tensor's native scale (no
clamping), since out-of-range LR values are legitimate for this task.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OODVariantSpec:
    name: str
    sigma: float = 0.0          # additive/speckle noise std
    blur_kernel: int = 5
    blur_sigma: float = 1.0


# Fixed severities. Chosen to sit clearly outside the training-time
# lr_gaussian_noise / lr_speckle_noise / lr_gaussian_blur ranges in
# configs/default.yaml (which top out around sigma=0.02 / 0.03 and
# blur sigma=0.8), so these variants genuinely probe OOD robustness
# rather than in-distribution augmentation the model has already seen.
OOD_VARIANTS: Dict[str, OODVariantSpec] = {
    "ood_gaussian": OODVariantSpec(name="ood_gaussian", sigma=0.05),
    "ood_speckle": OODVariantSpec(name="ood_speckle", sigma=0.08),
    "ood_blur": OODVariantSpec(name="ood_blur", blur_kernel=5, blur_sigma=1.5),
    "ood_mixed": OODVariantSpec(
        name="ood_mixed", sigma=0.04, blur_kernel=5, blur_sigma=1.2
    ),
}


def _seeded_generator(variant_name: str, sample_index: int, base_seed: int = 20240521) -> torch.Generator:
    """
    Builds a torch.Generator seeded deterministically from
    (base_seed, variant_name, sample_index) only -- independent of any
    global RNG state, epoch, worker id, or run order. This is what makes
    the OOD benchmark reproducible across checkpoints and machines.
    """
    # Simple, stable string->int hash (avoid Python's salted hash()).
    h = 0
    for ch in variant_name:
        h = (h * 131 + ord(ch)) % (2**31 - 1)
    seed = (base_seed + h * 1_000_003 + sample_index) % (2**31 - 1)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _gaussian_blur_deterministic(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    x: [1, H, W] (single-channel, no batch dim). Fixed Gaussian blur kernel,
    reflect padding. Purely deterministic given kernel_size/sigma -- no RNG
    involved, so no generator argument is needed here.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    xb = x.unsqueeze(0)  # [1,1,H,W]
    pad = kernel_size // 2
    xb = F.pad(xb, [pad, pad, pad, pad], mode="reflect")
    xb = F.conv2d(xb, kernel_2d.to(dtype=xb.dtype))
    return xb.squeeze(0)


def apply_ood_variant(lr: torch.Tensor, variant_name: str, sample_index: int) -> torch.Tensor:
    """
    Applies a single named OOD variant to one LR tensor [1, H, W].

    lr: the *existing* organizer-provided NoisyLR tensor (already whatever
        degradation the organizers applied) -- this function stacks a
        deterministic extra perturbation on top of it. GT is never touched
        and is not passed in here at all.
    variant_name: one of OOD_VARIANTS keys.
    sample_index: stable index of this sample within the validation set
        (e.g. its position in the sorted val stem list), used purely to
        seed the RNG deterministically -- not used for any lookup.

    Returns a new tensor; does not modify `lr` in place.
    """
    if variant_name not in OOD_VARIANTS:
        raise ValueError(f"Unknown OOD variant '{variant_name}'. Available: {list(OOD_VARIANTS)}")

    spec = OOD_VARIANTS[variant_name]
    gen = _seeded_generator(variant_name, sample_index)
    out = lr.clone()

    if variant_name == "ood_gaussian":
        noise = torch.randn(out.shape, generator=gen, dtype=torch.float32)
        out = out + noise * spec.sigma

    elif variant_name == "ood_speckle":
        noise = torch.randn(out.shape, generator=gen, dtype=torch.float32)
        out = out + out * noise * spec.sigma

    elif variant_name == "ood_blur":
        out = _gaussian_blur_deterministic(out, kernel_size=spec.blur_kernel, sigma=spec.blur_sigma)

    elif variant_name == "ood_mixed":
        # Deterministic fixed order: blur -> additive Gaussian -> speckle.
        out = _gaussian_blur_deterministic(out, kernel_size=spec.blur_kernel, sigma=spec.blur_sigma)
        noise1 = torch.randn(out.shape, generator=gen, dtype=torch.float32)
        out = out + noise1 * spec.sigma
        noise2 = torch.randn(out.shape, generator=gen, dtype=torch.float32)
        out = out + out * noise2 * (spec.sigma * 0.5)

    else:  # pragma: no cover - guarded by the ValueError above
        raise AssertionError(f"Unhandled OOD variant: {variant_name}")

    return out


def variant_names():
    return list(OOD_VARIANTS.keys())
