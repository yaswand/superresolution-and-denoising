import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


def gaussian_kernel(window_size=11, sigma=1.5, channels=1):
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.kernel = gaussian_kernel(window_size, sigma)

    def forward(self, pred, target):
        device = pred.device
        dtype = pred.dtype
        kernel = self.kernel.to(device=device, dtype=dtype)
        pad = self.window_size // 2

        batch_size = pred.shape[0]
        losses = []

        for i in range(batch_size):
            p = pred[i : i + 1]
            t = target[i : i + 1]

            # Use the actual GT data range, matching validation SSIM
            dr = t.amax() - t.amin()
            dr = torch.clamp(dr, min=1e-6)

            c1 = (0.01 * dr) ** 2
            c2 = (0.03 * dr) ** 2

            mu_p = F.conv2d(p, kernel, padding=pad)
            mu_t = F.conv2d(t, kernel, padding=pad)

            mu_p_sq = mu_p**2
            mu_t_sq = mu_t**2
            mu_pt = mu_p * mu_t

            sigma_p_sq = F.conv2d(p * p, kernel, padding=pad) - mu_p_sq

            sigma_t_sq = F.conv2d(t * t, kernel, padding=pad) - mu_t_sq

            sigma_pt = F.conv2d(p * t, kernel, padding=pad) - mu_pt

            numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)

            denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)

            ssim_map = numerator / denominator

            losses.append(1.0 - ssim_map.mean())

        return torch.stack(losses).mean()


class RestorationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        loss_cfg = cfg.get("loss", {})
        self.pixel_weight = loss_cfg.get("weights", {}).get("pixel", 1.0)
        self.ssim_weight = loss_cfg.get("weights", {}).get("ssim", 0.2)

        eps = loss_cfg.get("charbonnier_eps", 1e-3)
        self.pixel_loss = CharbonnierLoss(eps) if loss_cfg.get("pixel_loss") == "charbonnier" else nn.L1Loss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target):
        loss = 0.0
        if self.pixel_weight > 0:
            loss += self.pixel_weight * self.pixel_loss(pred, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(pred, target)
        return loss
