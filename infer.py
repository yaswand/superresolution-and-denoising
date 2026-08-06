import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataset import IMAGE_EXTENSIONS, _load_array
from utils import load_config
from models import build_model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def process_image(model, img_tensor, tile_size, overlap, scale):
    _, _, h, w = img_tensor.shape
    if tile_size is None or (h <= tile_size and w <= tile_size):
        return model(img_tensor)

    # Tile fallback for massive images to prevent OOM
    out_h, out_w = h * scale, w * scale
    out_tensor = torch.zeros((1, 1, out_h, out_w), device=img_tensor.device)
    count_tensor = torch.zeros((1, 1, out_h, out_w), device=img_tensor.device)

    stride = tile_size - overlap
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            t = min(i + tile_size, h)
            l = min(j + tile_size, w)
            b = i if t - i == tile_size else max(0, t - tile_size)
            r = j if l - j == tile_size else max(0, l - tile_size)

            tile = img_tensor[:, :, b:t, r:l]
            pred_tile = model(tile)

            out_tensor[:, :, b * scale : t * scale, r * scale : l * scale] += pred_tile
            count_tensor[:, :, b * scale : t * scale, r * scale : l * scale] += 1

    return out_tensor / count_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("cfg", load_config("configs/default.yaml"))

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    use_amp = device.type == "cuda" and cfg.get("inference", {}).get("amp", True)
    tile_size = cfg.get("inference", {}).get("tile_size", None)
    overlap = cfg.get("inference", {}).get("tile_overlap", 32)
    scale = cfg["model"]["scale"]

    valid_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(IMAGE_EXTENSIONS + (".npy",))]

    for fname in valid_files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        arr = _load_array(in_path)
        img_t = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = process_image(model, img_t, tile_size, overlap, scale)

        pred_np = pred.squeeze().cpu().numpy()

        # Clamp ONLY for image formats, allow out-of-bounds for raw .npy scoring.
        if not fname.lower().endswith(".npy"):
            pred_np = np.clip(pred_np, 0.0, 1.0)
            Image.fromarray((pred_np * 255.0).astype(np.uint8)).save(out_path)
        else:
            np.save(out_path, pred_np)

        print(f"Processed: {fname} -> {out_path}")


if __name__ == "__main__":
    main()
