import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiModalSlopeDataset
from GatedGridUNet import GatedGridUNet, TopoLoss

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to dataset root (contains train/ and val/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=str(SCRIPT_DIR / "runs" / "framebeam"))
    parser.add_argument("--base-channels", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).expanduser().resolve()

    train_ds = MultiModalSlopeDataset(str(data_root), split="train", modalities=["grays", "depths", "curves"])
    val_ds = MultiModalSlopeDataset(str(data_root), split="val", modalities=["grays", "depths", "curves"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = GatedGridUNet(in_channels=3, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = TopoLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

        train_loss /= max(len(train_ds), 1)
        val_loss /= max(len(val_ds), 1)
        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_modalities": ["grays", "depths", "curves"],
                    "in_channels": 3,
                    "base_channels": args.base_channels,
                },
                best_path,
            )
            print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()
