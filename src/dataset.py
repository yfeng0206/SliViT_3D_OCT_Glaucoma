"""FairVision Glaucoma OCT dataset loader."""

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FairVisionGlaucomaDataset(Dataset):
    """
    Each .npz file contains:
        - 'oct_bscans': (200, 200, 200) uint8 array of OCT slices
        - 'glaucoma': binary label (0 or 1)
        - 'slo_fundus', 'race', 'male', 'hispanic': metadata (unused here)

    The dataset uniformly samples `num_slices` slices from the 200 available,
    resizes each to 256x256, converts to 3-channel RGB, and vertically tiles
    them into a single elongated image of shape 3 x (num_slices*256) x 256.
    """

    def __init__(
        self,
        data_dir: str,
        num_slices: int = 32,
        slice_size: int = 256,
        transform=None,
    ):
        """
        Args:
            data_dir: Path to directory containing .npz files
                      (e.g., .../Training, .../Validation, .../Test).
            num_slices: Number of slices to uniformly sample from the volume.
            slice_size: Spatial resolution to resize each slice to.
            transform: Optional additional torchvision transform applied to
                       the final tiled tensor.
        """
        self.data_dir = data_dir
        self.num_slices = num_slices
        self.slice_size = slice_size
        self.transform = transform

        self.file_paths = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
            ]
        )
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {data_dir}"
            )

        # Pre-compute the slice indices once (same for every sample).
        self.slice_indices = np.linspace(
            0, 199, self.num_slices
        ).astype(int)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.file_paths[idx])
        volume = data["oct_bscans"]  # (200, 200, 200)
        label = int(data["glaucoma"])

        slices_resized = []
        for si in self.slice_indices:
            # volume[si] is a 200x200 grayscale slice
            img = Image.fromarray(volume[si].astype(np.uint8), mode="L")
            img = img.resize(
                (self.slice_size, self.slice_size), Image.BILINEAR
            )
            # Convert to 3-channel RGB (replicates the gray channel)
            img = img.convert("RGB")
            # (H, W, 3) -> (3, H, W) float32 in [0, 1]
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # (3, 256, 256)
            slices_resized.append(arr)

        # Vertically tile: (3, num_slices*256, 256)
        tiled = np.concatenate(slices_resized, axis=1)
        image_tensor = torch.from_numpy(tiled)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, label_tensor
