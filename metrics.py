"""
metrics.py

PSNR and SSIM computed in de-normalized pixel space, using the GT's own
data range (max - min) rather than an assumed fixed range like 255 or 1.0,
since these are raw grayscale .npy arrays with an unknown native range.

Both metrics operate on batches of tensors shaped [B, 1, H, W].
"""

import torch
import torch.nn.functional as F


def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse per-image z-score normalization.
    mean/std are 1D tensors of shape [B], one value per image in the batch.
    """
    mean = mean.view(-1, 1, 1, 1)
    std = std.view(-1, 1, 1, 1)
    return tensor * std + mean


@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    """
    pred, target: [B, 1, H, W] in pixel space (already de-normalized).
    data_range: [B] tensor, per-image (max - min) of the GT image.
    Returns per-batch mean PSNR (scalar tensor).
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # [B]
    mse = torch.clamp(mse, min=1e-10)  # avoid log(0)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.mean()


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


@torch.no_grad()
def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Standard single-channel SSIM with a Gaussian window, per-image data_range.
    pred, target: [B, 1, H, W] in pixel space.
    data_range: [B] tensor, per-image (max - min) of the GT image.
    Returns per-batch mean SSIM (scalar tensor).
    """
    device, dtype = pred.device, pred.dtype
    kernel = _gaussian_kernel(window_size, sigma, device, dtype)
    pad = window_size // 2

    B = pred.shape[0]
    ssim_vals = []

    # Loop per-image because data_range (and hence C1/C2) differs per image.
    for i in range(B):
        p = pred[i:i + 1]
        t = target[i:i + 1]
        dr = data_range[i]

        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2

        mu_p = F.conv2d(p, kernel, padding=pad)
        mu_t = F.conv2d(t, kernel, padding=pad)

        mu_p_sq = mu_p ** 2
        mu_t_sq = mu_t ** 2
        mu_pt = mu_p * mu_t

        sigma_p_sq = F.conv2d(p * p, kernel, padding=pad) - mu_p_sq
        sigma_t_sq = F.conv2d(t * t, kernel, padding=pad) - mu_t_sq
        sigma_pt = F.conv2d(p * t, kernel, padding=pad) - mu_pt

        numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
        denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
        ssim_map = numerator / denominator

        ssim_vals.append(ssim_map.mean())

    return torch.stack(ssim_vals).mean()
