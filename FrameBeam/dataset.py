from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalSlopeDataset(Dataset):
    """
    Expected directory structure:
    root/
      train|val/
        grays/*.png
        depths/*.png
        curves/*.png
        masks/*.png
    """

    def __init__(self, root: str, split: str = "train", modalities=None):
        self.root = Path(root)
        self.split = split
        self.modalities = modalities or ["grays", "depths", "curves"]
        base_dir = self.root / split / self.modalities[0]
        self.files = sorted([p.name for p in base_dir.glob("*.png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        channels = []
        for m in self.modalities:
            img_path = self.root / self.split / m / name
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read modality file: {img_path}")
            channels.append(img)

        mask_path = self.root / self.split / "masks" / name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask file: {mask_path}")

        x = np.stack(channels, axis=0).astype(np.float32) / 255.0
        y = (mask.astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y), name
