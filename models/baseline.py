"""
model.py

Super-Resolution Residual U-Net for 2x upsampling + denoising.

Why not a plain U-Net:
  A standard U-Net keeps spatial resolution constant between input and output,
  so `output = input + residual` requires matching shapes. Here LR input is
  half the spatial size of GT output, so we need explicit upsampling built
  into the architecture, and the residual connection must be added AFTER
  upsampling the input to the target resolution.

Design:
  - Encoder operates at LR resolution (128x128 or 256x256 depending on split).
  - Bottleneck.
  - Decoder with skip connections, operating at LR resolution.
  - A dedicated 2x upsampling head at the END using PixelShuffle (sub-pixel
    convolution), which is preferred over plain bilinear+conv because it lets
    the network learn the upsampling kernel directly and avoids the smoothing/
    checkerboard tradeoffs of naive interpolation.
  - Residual connection: we bilinearly upsample the ORIGINAL input to HR size
    (cheap, fixed operation) and add it to the network's learned HR residual.
    This gives the network an easy identity/coarse-upsample baseline to refine,
    which stabilizes training (similar to residual learning in EDSR/VDSR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscale by 2 then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upscale by 2 (transpose conv), concat skip, then DoubleConv.

    Channel counts are passed explicitly rather than assumed symmetric,
    because in this architecture the skip tensors do not follow the
    "skip channels == in_ch // 2" convention of a plain U-Net (our skip
    connections come from shallower encoder stages with fewer channels
    than a symmetric design would produce).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        up_out_ch = in_ch // 2
        self.up = nn.ConvTranspose2d(in_ch, up_out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(up_out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle any off-by-one size mismatch from odd input dimensions.
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PixelShuffleUpsample(nn.Module):
    """
    2x upsampling head using sub-pixel convolution (PixelShuffle).
    Conv expands channels by scale^2, then PixelShuffle rearranges them
    into spatial resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale**2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return self.act(x)


class SRResidualUNet(nn.Module):
    """
    Super-Resolution Residual U-Net.

    Args:
        in_channels: 1 for grayscale.
        out_channels: 1 for grayscale.
        base_ch: number of channels at the first encoder level.
        scale: super-resolution factor (2 for this challenge).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_ch: int = 64, scale: int = 2):
        super().__init__()
        self.scale = scale

        # Encoder (operates at LR resolution)
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder (still at LR resolution, with skip connections).
        # Skip channel counts: x3 -> 4*base_ch, x2 -> 2*base_ch, x1 -> base_ch.
        self.up1 = Up(in_ch=base_ch * 16, skip_ch=base_ch * 4, out_ch=base_ch * 8)
        self.up2 = Up(in_ch=base_ch * 8, skip_ch=base_ch * 2, out_ch=base_ch * 4)
        self.up3 = Up(in_ch=base_ch * 4, skip_ch=base_ch * 1, out_ch=base_ch * 2)

        # Final feature refinement at LR resolution
        self.lr_out_conv = DoubleConv(base_ch * 2, base_ch)

        # Learned 2x upsampling to HR resolution
        self.sr_head = PixelShuffleUpsample(base_ch, base_ch, scale=scale)

        # Project to single-channel residual at HR resolution
        self.residual_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, H, W]  (LR input)

        # --- Encoder ---
        x1 = self.inc(x)  # [B, C,   H,   W]
        x2 = self.down1(x1)  # [B, 2C,  H/2, W/2]
        x3 = self.down2(x2)  # [B, 4C,  H/4, W/4]
        x4 = self.down3(x3)  # [B, 8C,  H/8, W/8]

        # --- Bottleneck ---
        xb = self.bottleneck(x4)  # [B, 16C, H/8, W/8]

        # --- Decoder (skip connections align by matching spatial resolution) ---
        d1 = self.up1(xb, x3)  # upsample H/8->H/4, concat skip x3 (H/4) -> [B, 8C, H/4, W/4]
        d2 = self.up2(d1, x2)  # upsample H/4->H/2, concat skip x2 (H/2) -> [B, 4C, H/2, W/2]
        d3 = self.up3(d2, x1)  # upsample H/2->H,   concat skip x1 (H)   -> [B, 2C, H,   W]

        feat = self.lr_out_conv(d3)  # [B, C, H, W]  back to LR spatial size

        # --- Learned upsampling to HR ---
        hr_feat = self.sr_head(feat)  # [B, C, 2H, 2W]
        residual = self.residual_conv(hr_feat)  # [B, 1, 2H, 2W]

        # --- Residual connection at HR resolution ---
        upsampled_input = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = upsampled_input + residual
        return out


if __name__ == "__main__":
    # Quick shape sanity check
    model = SRResidualUNet(in_channels=1, out_channels=1, base_ch=64, scale=2)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print("Input:", dummy.shape)
    print("Output:", out.shape)  # Expected: [2, 1, 256, 256]
