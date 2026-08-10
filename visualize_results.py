"""Create side-by-side LR input, prediction, and ground-truth result images."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def psnr_ssim(image: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Return PSNR and global SSIM for images in the nominal [0, 1] range."""
    image = np.clip(image, 0, 1).astype(np.float64)
    reference = np.clip(reference, 0, 1).astype(np.float64)
    mse = np.mean((image - reference) ** 2)
    psnr = float("inf") if mse == 0 else -10 * np.log10(mse)

    # SSIM constants for data_range=1, matching the standard formulation.
    c1, c2 = 0.01**2, 0.03**2
    mu_image, mu_reference = image.mean(), reference.mean()
    var_image = ((image - mu_image) ** 2).mean()
    var_reference = ((reference - mu_reference) ** 2).mean()
    covariance = ((image - mu_image) * (reference - mu_reference)).mean()
    ssim = ((2 * mu_image * mu_reference + c1) * (2 * covariance + c2)) / (
        (mu_image**2 + mu_reference**2 + c1) * (var_image + var_reference + c2)
    )
    return psnr, float(ssim)


def to_image(array: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """Convert a float image to an 8-bit display image, clipping only for viewing."""
    image = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8), mode="L")
    return image.resize(size, Image.Resampling.BICUBIC) if size else image


def labeled(image: Image.Image, label: str, metrics: tuple[float, float] | None = None) -> Image.Image:
    lines = [label]
    if metrics is not None:
        psnr, ssim = metrics
        psnr_text = "inf" if np.isinf(psnr) else f"{psnr:.2f} dB"
        lines.append(f"PSNR: {psnr_text} | SSIM: {ssim:.4f}")
    header_height = 14 * len(lines) + 8
    canvas = Image.new("L", (image.width, image.height + header_height), color=255)
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((6, 4 + 14 * index), line, fill=0)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LR / prediction / GT comparison PNGs.")
    parser.add_argument("--lr", default="train/NoisyLR", help="Directory of low-resolution .npy inputs")
    parser.add_argument("--predictions", required=True, help="Directory of predicted .npy images")
    parser.add_argument("--gt", default="train/GT", help="Directory of ground-truth .npy images")
    parser.add_argument("--output", default="visualizations", help="Output directory for comparison PNGs")
    parser.add_argument("--limit", type=int, default=8, help="Number of random comparisons (0 means all)")
    parser.add_argument("--seed", type=int, default=42, help="Random-sampling seed")
    args = parser.parse_args()

    lr_dir, pred_dir, gt_dir, output = map(Path, (args.lr, args.predictions, args.gt, args.output))
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for pred_path in pred_dir.glob("*.npy"):
        lr_path, gt_path = lr_dir / pred_path.name, gt_dir / pred_path.name
        if lr_path.exists() and gt_path.exists():
            candidates.append(pred_path)
    if not candidates:
        raise FileNotFoundError("No matching .npy files were found across LR, predictions, and GT directories.")
    if args.limit:
        candidates = random.Random(args.seed).sample(candidates, min(args.limit, len(candidates)))

    created = 0
    for pred_path in candidates:
        lr_path, gt_path = lr_dir / pred_path.name, gt_dir / pred_path.name
        lr, pred, gt = (np.load(path).astype(np.float32) for path in (lr_path, pred_path, gt_path))
        height, width = gt.shape[-2:]
        lr_upscaled = np.asarray(to_image(lr, (width, height)), dtype=np.float32) / 255
        panels = [
            labeled(to_image(lr, (width, height)), "LR input (upscaled)", psnr_ssim(lr_upscaled, gt)),
            labeled(to_image(pred, (width, height)), "Prediction", psnr_ssim(pred, gt)),
            labeled(to_image(gt, (width, height)), "Ground truth (reference)", psnr_ssim(gt, gt)),
        ]
        comparison = Image.new("L", (sum(p.width for p in panels), panels[0].height), color=255)
        x = 0
        for panel in panels:
            comparison.paste(panel, (x, 0))
            x += panel.width
        comparison.save(output / f"{pred_path.stem}.png")
        created += 1
    print(f"Created {created} comparison image(s) in: {output}")


if __name__ == "__main__":
    main()
