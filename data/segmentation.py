from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.preprocessing import (
    get_segmentation_train_transform,
    get_segmentation_val_transform,
    get_segmentation_train_transform_dino,
    get_segmentation_val_transform_dino,
)


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    name: str
    source: str


class GenericSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        transform=None,
        mask_threshold: float = 0.5,
    ):
        if len(samples) == 0:
            raise ValueError("SegmentationDataset requires at least one sample")
        self.samples = list(samples)
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            with Image.open(path) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Failed to open image file {path}: {e}")

    def _load_mask(self, path: Path) -> np.ndarray:
        # Masks are grayscale; make sure to convert and cast to float.
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")
        try:
            with Image.open(path) as mask_img:
                return np.array(mask_img.convert("L"), dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to open mask file {path}: {e}")

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask
        else:
            mask_tensor = torch.from_numpy(np.asarray(mask))

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]

        mask_tensor = (mask_tensor > self.mask_threshold).float()
        return image, mask_tensor


def _drive_mask_name(image_stem: str, use_manual: bool) -> str:
    prefix = image_stem.split("_")[0]
    if use_manual:
        return f"{prefix}_manual1.gif"
    return f"{image_stem}_mask.gif"


def discover_drive_samples(
    drive_root: str | Path,
    split: str = "training",
    use_manual: bool = True,
) -> List[SegmentationSample]:
    split_root = Path(drive_root) / split
    image_dir = split_root / "images"
    manual_dir = split_root / ("1st_manual" if use_manual else "mask")
    samples: List[SegmentationSample] = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        mask_name = _drive_mask_name(stem, use_manual)
        mask_path = manual_dir / mask_name
        if not mask_path.exists():
            continue
        samples.append(
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_path,
                name=stem,
                source="drive",
            )
        )
    return samples


def discover_kvasir_samples(kvasir_root: str | Path) -> List[SegmentationSample]:
    root = Path(kvasir_root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    samples: List[SegmentationSample] = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        samples.append(
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_path,
                name=image_path.stem,
                source="kvasir",
            )
        )
    return samples


def _split_samples(
    samples: Sequence[SegmentationSample],
    val_split: float,
    seed: int,
) -> Tuple[List[SegmentationSample], List[SegmentationSample]]:
    if len(samples) <= 1 or val_split <= 0:
        return list(samples), []

    val_count = int(len(samples) * val_split)
    val_count = min(max(val_count, 1), len(samples) - 1)
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    val_indices = set(indices[:val_count])
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]
    return train_samples, val_samples


class MedTokenSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        drive_root: Optional[str] = None,
        kvasir_root: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: int | Tuple[int, int] = 512,
        drive_val_split: float = 0.2,
        kvasir_val_split: float = 0.1,
        seed: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        preprocessing: str = "nnunet",
    ):
        super().__init__()
        self.drive_root = drive_root
        self.kvasir_root = kvasir_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.drive_val_split = drive_val_split
        self.kvasir_val_split = kvasir_val_split
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.preprocessing = preprocessing

        if drive_root is None and kvasir_root is None:
            raise ValueError("At least one of drive_root or kvasir_root must be provided")

        if drive_root is not None and kvasir_root is not None:
            raise ValueError("Only one of drive_root or kvasir_root can be provided at a time")

    def setup(self, stage: Optional[str] = None):
        if self.preprocessing == "dino":
            train_transform = get_segmentation_train_transform_dino(self.image_size)
            val_transform = get_segmentation_val_transform_dino(self.image_size)
        else:
            train_transform = get_segmentation_train_transform(self.image_size)
            val_transform = get_segmentation_val_transform(self.image_size)

        train_datasets: List[Dataset] = []
        val_datasets: List[Dataset] = []
        seed = self.seed


        if self.drive_root is not None:
            drive_samples = discover_drive_samples(self.drive_root, split="training", use_manual=True)
            drive_train, drive_val = _split_samples(drive_samples, self.drive_val_split, seed)
            if drive_train:
                train_datasets.append(GenericSegmentationDataset(drive_train, transform=train_transform))
            if drive_val:
                val_datasets.append(GenericSegmentationDataset(drive_val, transform=val_transform))
            # seed += 1

        if self.kvasir_root is not None:
            kvasir_samples = discover_kvasir_samples(self.kvasir_root)
            kvasir_train, kvasir_val = _split_samples(kvasir_samples, self.kvasir_val_split, seed)
            if kvasir_train:
                train_datasets.append(GenericSegmentationDataset(kvasir_train, transform=train_transform))
            if kvasir_val:
                val_datasets.append(GenericSegmentationDataset(kvasir_val, transform=val_transform))

        if not train_datasets:
            raise RuntimeError("No training samples were discovered for the configured datasets")

        self.train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        self.val_dataset = None
        if val_datasets:
            self.val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        multiprocessing_context = "spawn" if self.num_workers > 0 else None
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        multiprocessing_context = "spawn" if self.num_workers > 0 else None
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )
