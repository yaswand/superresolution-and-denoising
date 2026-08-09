"""
lpips_metric.py

Evaluation-only LPIPS wrapper for grayscale restoration output.

The `lpips` package (and its pretrained backbone weights) is an optional
dependency -- see requirements.txt. This module must never break training
or ID/OOD PSNR/SSIM if lpips isn't installed or its weights can't be
downloaded (e.g. no network on a training node): callers should catch the
RuntimeError from get_lpips_model() / LPIPSMetric.__init__ and simply skip
LPIPS for that run.

Grayscale handling (standard approach, per task brief):
    1. Clamp prediction and GT to [0, 1] (LPIPS backbones are trained on
       natural images in that range; this clamp is metric-only and never
       applied to the tensors used for the forward pass or PSNR/SSIM).
    2. Repeat the single channel to 3 channels.
    3. Map [0, 1] -> [-1, 1] (the range lpips.LPIPS expects).
    4. Run the LPIPS backbone.

This clamp-repeat-rescale convention is made explicit and centralized here
so both the training-time validate() path (if ever enabled) and the
standalone evaluate.py path use exactly the same convention.
"""

from typing import Optional

import torch
import torch.nn as nn


class LPIPSMetric:
    """
    Thin, lazily-initialized wrapper around the `lpips` package.

    Usage:
        metric = LPIPSMetric(net="alex", device=device)  # raises RuntimeError if unavailable
        score = metric(pred, gt)  # pred, gt: [B, 1, H, W], native scale
    """

    def __init__(self, net: str = "alex", device: Optional[torch.device] = None):
        try:
            import lpips  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "The `lpips` package is not installed. Install it with `pip install lpips` "
                "to enable LPIPS evaluation, or set metrics.compute_lpips: false / omit "
                "--lpips on evaluate.py to skip it."
            ) from e

        self.device = device if device is not None else torch.device("cpu")
        try:
            self._model: nn.Module = lpips.LPIPS(net=net).to(self.device)
        except Exception as e:
            # Typically a weight-download failure (no network on this node).
            raise RuntimeError(
                f"Failed to initialize LPIPS backbone '{net}' (possibly no network access "
                f"to fetch pretrained weights). Original error: {e}"
            ) from e
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _prepare(x: torch.Tensor) -> torch.Tensor:
        """
        [B, 1, H, W] native-scale tensor -> [B, 3, H, W] in [-1, 1], per the
        standard LPIPS-on-grayscale convention described in the module
        docstring. Metric-only clamp; does not mutate the input tensor.
        """
        x = torch.clamp(x, 0.0, 1.0)
        x = x.repeat(1, 3, 1, 1)
        x = x * 2.0 - 1.0
        return x

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred3 = self._prepare(pred).to(self.device)
        target3 = self._prepare(target).to(self.device)
        dist = self._model(pred3, target3)
        return dist.mean().item()


def try_build_lpips(net: str = "alex", device: Optional[torch.device] = None) -> Optional[LPIPSMetric]:
    """
    Best-effort constructor: returns an LPIPSMetric, or None (with a printed
    warning) if the `lpips` package / weights aren't available. Callers
    should treat None as "skip LPIPS for this run" rather than crashing.
    """
    try:
        return LPIPSMetric(net=net, device=device)
    except RuntimeError as e:
        print(f"[lpips_metric] LPIPS unavailable, skipping: {e}")
        return None
