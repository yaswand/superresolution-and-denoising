# Semiconductor joint denoising + super-resolution

The supplied KLA data contains 3,200 paired float32 `.npy` images: GT is 256×256 in `[0,1]`; NoisyLR is 128×128 and is intentionally not clipped because sampled values extend below 0 and above 1. The default 7.36M-parameter model predicts a residual over bicubic interpolation, uses shifted window attention and PixelShuffle, and has no GAN component.

```powershell
python train.py --config config.yaml
python infer.py --checkpoint checkpoints/best_ssim.pt --input train/NoisyLR --output predictions
python evaluate.py --predictions predictions --gt train/GT --report report.csv

python visualize_results.py --predictions predictions --output visualizations
python visualize_results.py --predictions predictions --output visualizations --seed 99
```

Training uses a Charbonnier warmup followed by Charbonnier + MS-SSIM + Sobel gradient loss. Pretrained VGG perceptual loss is available but disabled by default to avoid an implicit weight download; enable it only after confirming locally cached ImageNet weights and validating no inspection artifacts. Validation reports a complexity split based on GT edge density. Filename-only data has no structure metadata, so the default deterministic split is a held-out image split; set `split_mode: prefix_group` if filenames are changed to encode structure families.
