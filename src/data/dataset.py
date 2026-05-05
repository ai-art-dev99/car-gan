"""
Data loading utilities.

Supports:
  - Stanford Cars dataset (download via Kaggle or torchvision)
  - Any folder of images (ImageFolder-style)
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int = 64) -> transforms.Compose:
    """Standard augmentation pipeline for GAN training."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.12)),   # slightly larger then crop
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
    ])


def get_inference_transforms(image_size: int = 64) -> transforms.Compose:
    """No augmentation – for evaluation or reference images."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_dataloader(
    data_path: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader from a flat image folder.

    Expected structure:
        data_path/
          class_a/  ← can be a single dummy class "cars/"
            img001.jpg
            img002.jpg
            ...

    Stanford Cars tip: point data_path at the 'cars_train' directory;
    it already has per-class sub-folders which are ignored by GAN training.
    """
    dataset = datasets.ImageFolder(
        root=str(data_path),
        transform=get_transforms(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # avoids batch-norm issues with small last batches
    )
    print(f"[Data] {len(dataset):,} images | {len(loader):,} batches | batch_size={batch_size}")
    return loader


# ──────────────────────────────────────────
# Helper: download Stanford Cars via Kaggle
# ──────────────────────────────────────────
def download_stanford_cars(dest: str = "./data") -> None:
    """
    Downloads the Stanford Cars dataset using the Kaggle API.

    Prerequisites:
        pip install kaggle
        Place kaggle.json in ~/.kaggle/
    Run:
        python -c "from src.data.dataset import download_stanford_cars; download_stanford_cars()"
    """
    import subprocess, sys, os
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", "jutrera/stanford-car-dataset-by-classes-folder",
        "-p", str(dest_path), "--unzip",
    ]
    print("[Data] Downloading Stanford Cars dataset via Kaggle...")
    subprocess.run(cmd, check=True)
    print(f"[Data] Done. Dataset saved to {dest_path}")
