"""Datasets and augmentation for TopMiner training.

Two data sources:
  - FolderRealFakeDataset: data_root/real/*.{jpg,png,...} + data_root/fake/*.{...}
  - HFRealFakeDataset: any HuggingFace dataset with image and label fields

Augmentation: paper Section 3.2.3, 4 difficulty levels.
  0 - CenterCrop + Resize only
  1 - + RandomRotation(20 deg), RandomResizedCrop, H/V flips
  2 - + ColorJitter (saturation/contrast), random JPEG compression
  3 - + GaussianBlur, Gaussian noise
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class RandomJPEGCompression:
    """Re-encode the image as JPEG at a random quality."""

    def __init__(self, quality_range=(40, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > self.p:
            return img
        q = int(torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianNoise:
    """Additive Gaussian noise on tensors in [0, 1]."""

    def __init__(self, std_range=(0.0, 0.05), p: float = 0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        std = float(torch.empty(1).uniform_(*self.std_range).item())
        return (tensor + std * torch.randn_like(tensor)).clamp_(0.0, 1.0)


def build_transform(image_size: int = 380, level: int = 2, train: bool = True) -> Callable:
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if not train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    pil_steps: list = []
    if level >= 1:
        pil_steps += [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    else:
        pil_steps += [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
        ]

    if level >= 2:
        pil_steps += [
            transforms.ColorJitter(saturation=(0.5, 1.5), contrast=(0.5, 1.5)),
            RandomJPEGCompression(),
        ]

    tensor_steps: list = [transforms.ToTensor()]
    if level >= 3:
        tensor_steps += [
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            RandomGaussianNoise(),
        ]
    tensor_steps += [normalize]

    return transforms.Compose(pil_steps + tensor_steps)


def _scan_folder(root: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for label_dir, label in (("real", 0), ("fake", 1)):
        d = root / label_dir
        if not d.is_dir():
            raise FileNotFoundError(f"Expected subdirectory not found: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((str(p), label))
    if not items:
        raise RuntimeError(f"No images found under {root}")
    return items


class FolderRealFakeDataset(Dataset):
    """data_root/real/* + data_root/fake/*."""

    def __init__(self, data_root: str, transform: Callable):
        self.items = _scan_folder(Path(data_root))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _image_from_value(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
    return Image.fromarray(np.array(value)).convert("RGB")


class _SubsetWithTransform(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform: Callable):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class _MemoryRowsDataset(Dataset):
    def __init__(self, rows: list[tuple[object, int]], transform: Callable):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        image_value, label = self.rows[idx]
        return self.transform(_image_from_value(image_value)), label


def make_folder_split(
    data_root: str,
    val_split: float,
    train_transform: Callable,
    eval_transform: Callable,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Split a real/fake folder into train + val with separate transforms."""
    items = _scan_folder(Path(data_root))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(items))
    n_val = max(1, int(len(items) * val_split))
    val_idx = set(int(i) for i in perm[:n_val])
    train_items = [items[i] for i in range(len(items)) if i not in val_idx]
    val_items = [items[i] for i in range(len(items)) if i in val_idx]
    return (
        _SubsetWithTransform(train_items, train_transform),
        _SubsetWithTransform(val_items, eval_transform),
    )


def make_parquet_split(
    parquet_path: str,
    val_split: float,
    train_transform: Callable,
    eval_transform: Callable,
    label_field: str = "label",
    image_field: str = "image",
    label_real_value: str | int = "real",
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load a local OpenFake-style parquet shard and split it for smoke tests."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("Parquet mode needs `pip install pyarrow`.") from e

    table = pq.read_table(parquet_path)
    if max_samples is not None:
        table = table.slice(0, min(max_samples, table.num_rows))

    rows = [
        (row[image_field], _label_to_binary(row[label_field], label_real_value))
        for row in table.to_pylist()
    ]
    if len(rows) < 2:
        raise RuntimeError("Need at least two parquet rows for a train/val split")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rows))
    n_val = max(1, int(len(rows) * val_split))
    val_idx = set(int(i) for i in perm[:n_val])
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
    return _MemoryRowsDataset(train_rows, train_transform), _MemoryRowsDataset(val_rows, eval_transform)


class HFRealFakeDataset(Dataset):
    """Wrap any HuggingFace dataset with image and label fields.

    Real maps to 0, fake maps to 1. OpenFake uses string labels ("real"/"fake"),
    while many smaller datasets use numeric labels, so both are supported.
    """

    def __init__(
        self,
        dataset_id: str,
        config: str | None = None,
        split: str = "train",
        transform: Callable | None = None,
        image_field: str = "image",
        label_field: str = "label",
        label_real_value: str | int = 0,
        max_samples: int | None = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("HuggingFace dataset mode needs `pip install datasets`.") from e

        if config:
            self.dataset = load_dataset(dataset_id, config, split=split)
        else:
            self.dataset = load_dataset(dataset_id, split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        self.transform = transform
        self.image_field = image_field
        self.label_field = label_field
        self.label_real_value = label_real_value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        img = _image_from_value(row[self.image_field])
        if self.transform:
            img = self.transform(img)
        label = _label_to_binary(row[self.label_field], self.label_real_value)
        return img, label


def _label_to_binary(raw_label, real_value: str | int) -> int:
    """Return 0 for real, 1 for fake/synthetic."""
    if isinstance(raw_label, str):
        value = raw_label.strip().lower()
        real = str(real_value).strip().lower()
        if value == real or value in {"real", "authentic", "human"}:
            return 0
        if value in {"fake", "synthetic", "ai", "generated", "ai-generated"}:
            return 1
        raise ValueError(f"Unrecognized string label: {raw_label!r}")

    try:
        return 0 if int(raw_label) == int(real_value) else 1
    except Exception as e:
        raise ValueError(f"Unrecognized label value: {raw_label!r}") from e
