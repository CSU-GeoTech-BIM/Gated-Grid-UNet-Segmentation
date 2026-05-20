import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from GatedGridUNet import GatedGridUNet

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--sample-dir", type=str, required=True, help="Single-sample directory containing img/depth/curve")
    parser.add_argument("--out-path", type=str, default=str(SCRIPT_DIR / "prediction.png"))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path).expanduser().resolve()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location=device)
    in_channels = ckpt.get("in_channels", 3)
    base_channels = ckpt.get("base_channels", 32)
    model = GatedGridUNet(in_channels=in_channels, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    img = read_gray(sample_dir / "img.png")
    depth = read_gray(sample_dir / "depth.png")
    curve = read_gray(sample_dir / "curve.png")

    h, w = img.shape
    x = np.stack([img, depth, curve], axis=0).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

    pred = (prob > 0.5).astype(np.uint8) * 255
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_path), pred)
    print(f"Saved prediction to: {out_path}")


if __name__ == "__main__":
    main()
