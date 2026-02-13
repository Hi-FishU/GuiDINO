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
    get_segmentation_val_transform_fullres,
    get_segmentation_train_transform_dino,
    get_segmentation_val_transform_dino,
    get_segmentation_val_transform_dino_fullres,
    get_segmentation_train_transform_dino_strong,
)


def _resolve_patch_size(patch_size: int | Tuple[int, int] | None) -> Optional[Tuple[int, int]]:
    if patch_size is None:
        return None
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2:
            raise ValueError("patch_size must be an int or (height, width)")
        return int(patch_size[0]), int(patch_size[1])
    size = int(patch_size)
    return size, size


def _pad_to_size(image: np.ndarray, mask: np.ndarray, target_h: int, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if image.ndim == 2:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    else:
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
    mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
    return image, mask


def _sample_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int],
    oversample_foreground_prob: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    target_h, target_w = patch_size
    image, mask = _pad_to_size(image, mask, target_h, target_w)
    h, w = image.shape[:2]
    max_y = h - target_h
    max_x = w - target_w

    use_foreground = False
    if oversample_foreground_prob > 0:
        if rng.random() < oversample_foreground_prob:
            use_foreground = np.any(mask > 0)

    if use_foreground:
        ys, xs = np.where(mask > 0)
        idx = rng.integers(0, len(ys))
        center_y = int(ys[idx])
        center_x = int(xs[idx])
        start_y = min(max(center_y - target_h // 2, 0), max_y)
        start_x = min(max(center_x - target_w // 2, 0), max_x)
    else:
        start_y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        start_x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0

    end_y = start_y + target_h
    end_x = start_x + target_w
    image = image[start_y:end_y, start_x:end_x]
    mask = mask[start_y:end_y, start_x:end_x]
    return image, mask


@dataclass(frozen=True)
class SegmentationSample:
    image_path: Path
    mask_path: Path
    name: str
    source: str


@dataclass(frozen=True)
class SynapseVolumeSample:
    image_path: Path
    mask_path: Path
    name: str
    source: str = "synapse"


class GenericSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SegmentationSample],
        transform=None,
        mask_threshold: float = 0.5,
        patch_size: int | Tuple[int, int] | None = None,
        oversample_foreground_prob: float = 0.0,
    ):
        if len(samples) == 0:
            raise ValueError("SegmentationDataset requires at least one sample")
        self.samples = list(samples)
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.patch_size = _resolve_patch_size(patch_size)
        self.oversample_foreground_prob = float(oversample_foreground_prob)
        self.rng = np.random.default_rng()

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

        if self.patch_size is not None:
            image, mask = _sample_patch(
                image,
                mask,
                self.patch_size,
                self.oversample_foreground_prob,
                self.rng,
            )

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


class SynapseSliceDataset(Dataset):
    def __init__(
        self,
        volumes: Sequence[SynapseVolumeSample],
        transform=None,
        include_empty: bool = False,
        to_rgb: bool = False,
        cache_data: bool = False,
        target_spacing: Tuple[float, float, float] | None = None,
        crop_nonzero: bool = True,
        zscore: bool = True,
        patch_size: int | Tuple[int, int] | None = None,
        oversample_foreground_prob: float = 0.0,
    ):
        if len(volumes) == 0:
            raise ValueError("SynapseSliceDataset requires at least one volume")
        self.volumes = list(volumes)
        self.transform = transform
        self.include_empty = include_empty
        self.to_rgb = to_rgb
        self.cache_data = cache_data
        self.target_spacing = target_spacing
        self.crop_nonzero = crop_nonzero
        self.zscore = zscore
        self.patch_size = _resolve_patch_size(patch_size)
        self.oversample_foreground_prob = float(oversample_foreground_prob)
        self.rng = np.random.default_rng()
        self._volume_cache: dict[Path, np.ndarray] = {}
        self._label_cache: dict[Path, np.ndarray] = {}
        self.samples: List[Tuple[SynapseVolumeSample, int]] = []

        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "SynapseSliceDataset requires nibabel. Install it with `pip install nibabel`."
            ) from exc

        for volume in self.volumes:
            image_data, mask_data = self._load_case(volume)
            if mask_data.ndim < 3:
                raise ValueError(f"Expected 3D mask, got shape {mask_data.shape} for {volume.mask_path}")
            slice_axis = mask_data.ndim - 1
            if self.include_empty:
                slice_indices = range(mask_data.shape[slice_axis])
            else:
                slice_indices = [
                    idx
                    for idx in range(mask_data.shape[slice_axis])
                    if np.any(mask_data.take(idx, axis=slice_axis))
                ]
            for slice_idx in slice_indices:
                self.samples.append((volume, int(slice_idx)))

        if not self.samples:
            raise ValueError("No Synapse slices were found (check include_empty or labels).")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_case(self, volume: SynapseVolumeSample) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_data:
            cached_image = self._volume_cache.get(volume.image_path)
            cached_mask = self._label_cache.get(volume.mask_path)
            if cached_image is not None and cached_mask is not None:
                return cached_image, cached_mask
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "SynapseSliceDataset requires nibabel. Install it with `pip install nibabel`."
            ) from exc
        try:
            from scipy.ndimage import zoom
        except ImportError as exc:
            raise ImportError(
                "SynapseSliceDataset requires scipy for resampling. Install it with `pip install scipy`."
            ) from exc

        image_obj = nib.load(str(volume.image_path))
        mask_obj = nib.load(str(volume.mask_path))
        image_data = image_obj.get_fdata().astype(np.float32)
        mask_data = mask_obj.get_fdata().astype(np.int16)

        if image_data.ndim < 3:
            raise ValueError(f"Expected 3D image, got shape {image_data.shape} for {volume.image_path}")

        if self.crop_nonzero:
            nonzero = image_data != 0
            if not np.any(nonzero):
                nonzero = mask_data > 0
            if np.any(nonzero):
                coords = np.array(np.where(nonzero))
                start = coords.min(axis=1)
                end = coords.max(axis=1) + 1
                slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
                image_data = image_data[slices]
                mask_data = mask_data[slices]

        if self.zscore:
            mask = image_data != 0
            if np.any(mask):
                mean = float(image_data[mask].mean())
                std = float(image_data[mask].std())
            else:
                mean = float(image_data.mean())
                std = float(image_data.std())
            image_data = (image_data - mean) / (std + 1e-8)

        if self.target_spacing is not None:
            current_spacing = image_obj.header.get_zooms()[: image_data.ndim]
            zoom_factors = [cs / ts for cs, ts in zip(current_spacing, self.target_spacing)]
            image_data = zoom(image_data, zoom=zoom_factors, order=1)
            mask_data = zoom(mask_data, zoom=zoom_factors, order=0)

        if self.cache_data:
            self._volume_cache[volume.image_path] = image_data
            self._label_cache[volume.mask_path] = mask_data
        return image_data, mask_data

    def __getitem__(self, idx: int):
        volume, slice_idx = self.samples[idx]
        image_vol, mask_vol = self._load_case(volume)
        mask_vol = mask_vol.astype(np.int16)
        slice_axis = image_vol.ndim - 1
        image = image_vol.take(slice_idx, axis=slice_axis)
        mask = mask_vol.take(slice_idx, axis=slice_axis)

        if self.patch_size is not None:
            image, mask = _sample_patch(
                image,
                mask,
                self.patch_size,
                self.oversample_foreground_prob,
                self.rng,
            )

        if self.to_rgb:
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            if image.ndim == 2:
                image = torch.from_numpy(image).unsqueeze(0).float()
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).long()

        if isinstance(mask, torch.Tensor):
            mask_tensor = mask
        else:
            mask_tensor = torch.from_numpy(np.asarray(mask))

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.long()
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


def discover_isic_samples(isic_root: str | Path) -> List[SegmentationSample]:
    root = Path(isic_root)
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
    layout_candidates = [
        (root / "images", root / "masks"),
        (root / "img", root / "label"),
    ]
    image_dir = root / "images"
    mask_dir = root / "masks"
    for candidate_image_dir, candidate_mask_dir in layout_candidates:
        if candidate_image_dir.exists() and candidate_mask_dir.exists():
            image_dir = candidate_image_dir
            mask_dir = candidate_mask_dir
            break

    mask_by_stem: dict[str, Path] = {}
    for mask_path in sorted(mask_dir.glob("*")):
        if not mask_path.is_file():
            continue
        if mask_path.suffix.lower() not in allowed_suffixes:
            continue
        mask_stem = mask_path.stem
        mask_by_stem[mask_stem] = mask_path
        if mask_stem.endswith("_segmentation"):
            base_stem = mask_stem[: -len("_segmentation")]
            mask_by_stem.setdefault(base_stem, mask_path)

    samples: List[SegmentationSample] = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in allowed_suffixes:
            continue
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        samples.append(
            SegmentationSample(
                image_path=image_path,
                mask_path=mask_path,
                name=image_path.stem,
                source="isic",
            )
        )
    return samples


def discover_synapse_volumes(synapse_root: str | Path) -> List[SynapseVolumeSample]:
    root = Path(synapse_root)
    image_dir = root / "averaged-training-images"
    mask_dir = root / "averaged-training-labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Synapse images directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Synapse labels directory not found: {mask_dir}")

    samples: List[SynapseVolumeSample] = []
    for image_path in sorted(image_dir.glob("*_avg.nii.gz")):
        if not image_path.is_file():
            continue
        stem = image_path.stem.replace(".nii", "")
        mask_path = mask_dir / f"{stem}_seg.nii.gz"
        if not mask_path.exists():
            continue
        samples.append(
            SynapseVolumeSample(
                image_path=image_path,
                mask_path=mask_path,
                name=stem,
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
        isic_root: Optional[str] = None,
        synapse_root: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: int | Tuple[int, int] = 512,
        drive_val_split: float = 0.2,
        kvasir_val_split: float = 0.1,
        isic_val_split: float = 0.1,
        synapse_val_split: float = 0.2,
        synapse_include_empty: bool = False,
        synapse_to_rgb: bool = False,
        synapse_cache: bool = False,
        synapse_target_spacing: Tuple[float, float, float] | None = None,
        synapse_crop_nonzero: bool = True,
        synapse_zscore: bool = True,
        seed: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        dataloader_mp_context: Optional[str] = None,
        preprocessing: str = "nnunet",
        patch_size: int | Tuple[int, int] | None = None,
        oversample_foreground_prob: float = 0.33,
        full_res_val_eval: bool = False,
    ):
        super().__init__()
        self.drive_root = drive_root
        self.kvasir_root = kvasir_root
        self.isic_root = isic_root
        self.synapse_root = synapse_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.drive_val_split = drive_val_split
        self.kvasir_val_split = kvasir_val_split
        self.isic_val_split = isic_val_split
        self.synapse_val_split = synapse_val_split
        self.synapse_include_empty = synapse_include_empty
        self.synapse_to_rgb = synapse_to_rgb
        self.synapse_cache = synapse_cache
        self.synapse_target_spacing = synapse_target_spacing
        self.synapse_crop_nonzero = synapse_crop_nonzero
        self.synapse_zscore = synapse_zscore
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.dataloader_mp_context = dataloader_mp_context
        self.preprocessing = preprocessing
        self.patch_size = patch_size
        self.oversample_foreground_prob = float(oversample_foreground_prob)
        self.full_res_val_eval = bool(full_res_val_eval)

        dataset_roots = [root is not None for root in (drive_root, kvasir_root, isic_root, synapse_root)]
        if sum(dataset_roots) == 0:
            raise ValueError("At least one of drive_root, kvasir_root, isic_root, or synapse_root must be provided")
        if sum(dataset_roots) > 1:
            raise ValueError("Only one of drive_root, kvasir_root, isic_root, or synapse_root can be provided at a time")

    def setup(self, stage: Optional[str] = None):
        train_transform_size = self.patch_size if self.patch_size is not None else self.image_size
        val_transform_size = self.image_size

        if self.preprocessing == "dino":
            train_transform = get_segmentation_train_transform_dino(train_transform_size)
            if self.full_res_val_eval:
                val_transform = get_segmentation_val_transform_dino_fullres()
            else:
                val_transform = get_segmentation_val_transform_dino(val_transform_size)
        elif self.preprocessing == "dino_strong":
            train_transform = get_segmentation_train_transform_dino_strong(train_transform_size)
            if self.full_res_val_eval:
                val_transform = get_segmentation_val_transform_dino_fullres()
            else:
                val_transform = get_segmentation_val_transform_dino(val_transform_size)
        else:
            train_transform = get_segmentation_train_transform(train_transform_size)
            if self.full_res_val_eval:
                val_transform = get_segmentation_val_transform_fullres()
            else:
                val_transform = get_segmentation_val_transform(val_transform_size)

        train_datasets: List[Dataset] = []
        val_datasets: List[Dataset] = []
        seed = self.seed


        if self.drive_root is not None:
            drive_samples = discover_drive_samples(self.drive_root, split="training", use_manual=True)
            drive_train, drive_val = _split_samples(drive_samples, self.drive_val_split, seed)
            if drive_train:
                train_datasets.append(
                    GenericSegmentationDataset(
                        drive_train,
                        transform=train_transform,
                        patch_size=self.patch_size,
                        oversample_foreground_prob=self.oversample_foreground_prob,
                    )
                )
            if drive_val:
                val_datasets.append(
                    GenericSegmentationDataset(
                        drive_val,
                        transform=val_transform,
                        patch_size=None,
                        oversample_foreground_prob=0.0,
                    )
                )
            # seed += 1

        if self.kvasir_root is not None:
            kvasir_samples = discover_kvasir_samples(self.kvasir_root)
            kvasir_train, kvasir_val = _split_samples(kvasir_samples, self.kvasir_val_split, seed)
            if kvasir_train:
                train_datasets.append(
                    GenericSegmentationDataset(
                        kvasir_train,
                        transform=train_transform,
                        patch_size=self.patch_size,
                        oversample_foreground_prob=self.oversample_foreground_prob,
                    )
                )
            if kvasir_val:
                val_datasets.append(
                    GenericSegmentationDataset(
                        kvasir_val,
                        transform=val_transform,
                        patch_size=None,
                        oversample_foreground_prob=0.0,
                    )
                )

        if self.isic_root is not None:
            isic_samples = discover_isic_samples(self.isic_root)
            isic_train, isic_val = _split_samples(isic_samples, self.isic_val_split, seed)
            if isic_train:
                train_datasets.append(
                    GenericSegmentationDataset(
                        isic_train,
                        transform=train_transform,
                        patch_size=self.patch_size,
                        oversample_foreground_prob=self.oversample_foreground_prob,
                    )
                )
            if isic_val:
                val_datasets.append(
                    GenericSegmentationDataset(
                        isic_val,
                        transform=val_transform,
                        patch_size=None,
                        oversample_foreground_prob=0.0,
                    )
                )

        if self.synapse_root is not None:
            synapse_volumes = discover_synapse_volumes(self.synapse_root)
            synapse_train, synapse_val = _split_samples(synapse_volumes, self.synapse_val_split, seed)
            if synapse_train:
                train_datasets.append(
                    SynapseSliceDataset(
                        synapse_train,
                        transform=train_transform,
                        include_empty=self.synapse_include_empty,
                        to_rgb=self.synapse_to_rgb,
                        cache_data=self.synapse_cache,
                        target_spacing=self.synapse_target_spacing,
                        crop_nonzero=self.synapse_crop_nonzero,
                        zscore=self.synapse_zscore,
                        patch_size=self.patch_size,
                        oversample_foreground_prob=self.oversample_foreground_prob,
                    )
                )
            if synapse_val:
                val_datasets.append(
                    SynapseSliceDataset(
                        synapse_val,
                        transform=val_transform,
                        include_empty=True,
                        to_rgb=self.synapse_to_rgb,
                        cache_data=self.synapse_cache,
                        target_spacing=self.synapse_target_spacing,
                        crop_nonzero=self.synapse_crop_nonzero,
                        zscore=self.synapse_zscore,
                        patch_size=None,
                        oversample_foreground_prob=0.0,
                    )
                )

        if not train_datasets:
            raise RuntimeError("No training samples were discovered for the configured datasets")

        self.train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        self.val_dataset = None
        if val_datasets:
            self.val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        multiprocessing_context = self.dataloader_mp_context if self.num_workers > 0 else None
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
        multiprocessing_context = self.dataloader_mp_context if self.num_workers > 0 else None
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(
            self.val_dataset,
            batch_size=1 if self.full_res_val_eval else self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )
