# Semiconductor Image Restoration Pipeline (KLA Challenge)

## Files

| File | Purpose |
|---|---|
| `verify_dims.py` | Run FIRST. Confirms GT/NoisyLR shapes, scale factor, and value ranges. |
| `dataset.py` | `SemiconductorSRDataset` + `make_train_val_split`. Per-image normalization, no clipping. |
| `model.py` | `SRResidualUNet`: Residual U-Net with a PixelShuffle 2x upsampling head. |
| `metrics.py` | PSNR / SSIM computed in de-normalized pixel space, per-image data range. |
| `train.py` | Full training loop: AdamW, L1 loss, cosine LR schedule, AMP, checkpointing. |

## Usage

```bash
# 1. Confirm dimensions and scale factor before touching the model
python verify_dims.py --root /path/to/train

# 2. Train (scale defaults to 2, matching the challenge brief)
python train.py --root /path/to/train --epochs 100 --batch_size 8 --scale 2

# Resume from a checkpoint
python train.py --root /path/to/train --resume ./checkpoints/last.pt
```

## Key design decisions

- **Per-image normalization, computed separately for LR and GT.** Speckle
  noise can push NoisyLR values outside the GT's range, so using one
  image's mean/std to normalize the other would bias the residual the
  network has to learn. Each tensor is normalized with its own statistics,
  and the GT's mean/std are carried through the batch so predictions can be
  de-normalized back to pixel space for PSNR/SSIM.
- **No clipping anywhere** in `dataset.py` or `model.py`, per the brief.
- **SRResidualUNet** encodes at LR resolution (3 down/up stages + bottleneck),
  then upsamples 2x using PixelShuffle (sub-pixel convolution) rather than
  plain bilinear + conv, since it lets the network learn the upsampling
  kernel directly. The residual connection adds a bilinearly-upsampled copy
  of the LR input to the network's learned HR residual, at HR resolution
  where the shapes actually match — this is why the previous
  `output = input + residual` failed (shapes didn't match) and needed the
  input pre-upsampled before the addition.
- **AMP and device selection are automatic.** `torch.cuda.is_available()`
  gates both `.to(device)` and AMP; the code never calls `.cuda()` directly,
  so it runs on your current CPU-only install without modification and will
  pick up CUDA automatically once you install the GPU build of PyTorch.
- **Checkpointing** saves both `last.pt` (for resuming) and `best.pt`
  (tracked by validation PSNR) every epoch.

## Sanity-tested

This pipeline was smoke-tested end-to-end on a synthetic 20-pair dataset
(64x64 GT / 32x32 NoisyLR with injected speckle+Gaussian noise) to confirm:
shapes flow correctly through the model (128->256 style upsampling), the
train/val loop runs without errors, PSNR/SSIM compute sensible values, and
checkpoints save correctly. Swap in your real 3200-pair dataset directly —
no code changes needed unless your actual resolution pair differs from
256<->512 or 128<->256 (in which case re-run `verify_dims.py` and pass
`--scale` accordingly).

## Next steps you may want to add

- Data augmentation (random flips/rotations — safe since they don't touch
  pixel statistics or introduce handcrafted denoising).
- TensorBoard or Weights & Biases logging.
- Test-time ensembling (flip/rotate averaging) for the final submission.
- If val PSNR plateaus, try increasing `base_ch` or adding perceptual loss
  (with caution — the brief only requires L1, and perceptual losses can
  hallucinate structure that may hurt on OOD semiconductor patterns).
