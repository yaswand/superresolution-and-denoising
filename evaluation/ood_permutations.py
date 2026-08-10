"""
ood_permutations.py

Order-generalization evaluation for the organizer-specified degradation
pipeline:

    S = speckle (multiplicative) noise
    G = additive Gaussian noise
    D = downsampling (GT resolution -> LR resolution, factor = scale)

The organizers explicitly state the pipeline is built from these three
operations but that the hidden test may apply them in *any* order, and
that the model should not be graded only on the one order it happened to
see in training. This module builds all 6 permutations of {S, G, D},
starting from the *clean GT* (not from the pre-existing NoisyLR, since D
must actually change resolution), and evaluates a given model against
each one.

This is a different axis from ood.py's ood_gaussian / ood_speckle /
ood_blur / ood_mixed variants:
  - ood.py perturbs the *existing* NoisyLR (already at LR resolution) and
    never re-derives it from GT, so it cannot express "downsampling
    happens after noise" vs "downsampling happens before noise" -- there
    is nothing left to downsample once you start from NoisyLR.
  - ood_permutations.py starts from GT (still at full resolution) and
    applies S, G, D in a specific order to *build* the LR input, so all
    6 orderings are actually expressible and distinct.

Determinism:
    Exactly the same philosophy as ood.py: every (order, sample_index)
    pair gets its own seeded generator, independent of wall-clock time,
    global RNG state, worker id, or run order, so results are
    reproducible and comparable across checkpoints/experiments.

Severities:
    Chosen to match ood.py's OOD_VARIANTS severities (sigma=0.05 additive
    Gaussian, sigma=0.08 speckle) so this evaluation sits at a comparable
    difficulty to your existing OOD suite, rather than introducing a new,
    incomparable severity scale.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Severities -- matched to ood.py's OOD_VARIANTS so results are comparable to
# your existing ood_gaussian / ood_speckle numbers.
# -----------------------------------------------------------------------------
GAUSSIAN_SIGMA = 0.05
SPECKLE_SIGMA = 0.08


@dataclass(frozen=True)
class PermutationSpec:
    label: str  # e.g. "P1"
    order: Tuple[str, str, str]  # e.g. ("S", "G", "D")


def _build_all_orders() -> List[PermutationSpec]:
    specs = []
    for i, order in enumerate(permutations(("S", "G", "D")), start=1):
        specs.append(PermutationSpec(label=f"P{i}", order=order))
    return specs


# Fixed, stable enumeration of all 6 orders. Index 0 -> P1, etc.
# P1 S->G->D, P2 S->D->G, P3 G->S->D, P4 G->D->S, P5 D->S->G, P6 D->G->S
ALL_PERMUTATIONS: List[PermutationSpec] = _build_all_orders()


def order_key(order: Tuple[str, str, str]) -> str:
    return "->".join(order)


def _seeded_generator(order_key_str: str, op: str, sample_index: int, base_seed: int = 20240521) -> torch.Generator:
    """
    Same deterministic-hash seeding scheme as ood.py's _seeded_generator,
    extended with the op letter (S/G) so that, within one order, the two
    noise-adding steps get independent (but still fully reproducible)
    random draws instead of reusing the same stream.
    """
    h = 0
    for ch in f"{order_key_str}|{op}":
        h = (h * 131 + ord(ch)) % (2**31 - 1)
    seed = (base_seed + h * 1_000_003 + sample_index) % (2**31 - 1)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _apply_speckle(x: torch.Tensor, order_key_str: str, sample_index: int) -> torch.Tensor:
    gen = _seeded_generator(order_key_str, "S", sample_index)
    noise = torch.randn(x.shape, generator=gen, dtype=torch.float32)
    return x + x * noise * SPECKLE_SIGMA


def _apply_gaussian(x: torch.Tensor, order_key_str: str, sample_index: int) -> torch.Tensor:
    gen = _seeded_generator(order_key_str, "G", sample_index)
    noise = torch.randn(x.shape, generator=gen, dtype=torch.float32)
    return x + noise * GAUSSIAN_SIGMA


def _apply_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    x: [1, H, W] at GT resolution. Returns [1, H/scale, W/scale].
    Uses area-averaging (equivalent to box-filter downsampling), a
    standard, deterministic choice with no RNG involved -- matches the
    typical LR-generation convention for super-resolution tasks.
    """
    xb = x.unsqueeze(0)  # [1, 1, H, W]
    out = F.interpolate(xb, scale_factor=1.0 / scale, mode="area")
    return out.squeeze(0)


_OP_FUNCS: Dict[str, Callable] = {
    "S": _apply_speckle,
    "G": _apply_gaussian,
    # "D" handled specially below since it needs `scale` and changes shape
}


def apply_permutation(
    gt: torch.Tensor,
    order: Tuple[str, str, str],
    scale: int,
    sample_index: int,
) -> torch.Tensor:
    """
    gt: [1, H, W] clean ground-truth tensor (native scale, untouched).
    order: e.g. ("S", "G", "D") -- degradation application order.
    scale: GT -> LR downsampling factor (matches the model's scale).
    sample_index: stable index of this sample in the val set, used for
        deterministic RNG seeding.

    Returns a new LR tensor [1, H/scale, W/scale]. Does not modify `gt`.
    """
    ok = order_key(order)
    out = gt.clone()

    for op in order:
        if op == "D":
            out = _apply_downsample(out, scale)
        else:
            out = _OP_FUNCS[op](out, ok, sample_index)

    return out


def permutation_labels() -> List[str]:
    return [spec.label for spec in ALL_PERMUTATIONS]


def permutation_order_for_label(label: str) -> Tuple[str, str, str]:
    for spec in ALL_PERMUTATIONS:
        if spec.label == label:
            return spec.order
    raise ValueError(f"Unknown permutation label '{label}'. Available: {permutation_labels()}")
