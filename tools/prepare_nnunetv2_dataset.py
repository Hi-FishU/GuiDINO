from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from data.segmentation import (
    SegmentationSample,
    discover_isic_samples,
    discover_isic_split_samples,
    discover_kvasir_samples,
    discover_tn3k_samples,
)


def _load_rgb_or_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"))
    return (arr > 0).astype(np.uint8)


def _save_channels_as_png(image: np.ndarray, stem: str, out_dir: Path) -> int:
    channels = image.shape[2]
    for c in range(channels):
        out_path = out_dir / f"{stem}_{c:04d}.png"
        Image.fromarray(image[..., c]).save(out_path)
    return channels


def _copy_cases(
    samples: Sequence[SegmentationSample],
    images_dir: Path,
    labels_dir: Path | None,
    prefix: str,
    start_index: int = 0,
    with_labels: bool = True,
) -> tuple[int, int]:
    max_channels = 1
    index = start_index
    for sample in samples:
        case_id = f"{prefix}_{index:04d}"
        image = _load_rgb_or_gray(sample.image_path)
        max_channels = max(max_channels, _save_channels_as_png(image, case_id, images_dir))
        if with_labels and labels_dir is not None:
            mask = _load_binary_mask(sample.mask_path)
            Image.fromarray(mask).save(labels_dir / f"{case_id}.png")
        index += 1
    return index, max_channels


def _resolve_samples(
    dataset: str,
    root: Path,
) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    key = dataset.lower()
    if key == "tn3k":
        return discover_tn3k_samples(root, split="trainval"), discover_tn3k_samples(root, split="test")
    if key == "kvasir":
        return discover_kvasir_samples(root), []
    if key == "isic":
        split_samples = discover_isic_split_samples(root)
        if split_samples is not None:
            return list(split_samples[0]), list(split_samples[1])
        return discover_isic_samples(root), []
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _write_dataset_json(
    dataset_dir: Path,
    dataset_name: str,
    max_channels: int,
    num_training: int,
) -> None:
    channel_names = {str(i): f"channel_{i}" for i in range(max_channels)}
    payload = {
        "name": dataset_name,
        "description": f"{dataset_name} converted for nnUNet v2 (2D natural images).",
        "tensorImageSize": "2D",
        "channel_names": channel_names,
        "labels": {"background": 0, "foreground": 1},
        "numTraining": int(num_training),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _prepare_layout(base_dir: Path, dataset_id: int, dataset_name: str, overwrite: bool) -> Path:
    dataset_dir = base_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dataset_dir} exists. Use --overwrite to rebuild or choose a different --dataset-id."
            )
        shutil.rmtree(dataset_dir)
    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MedToken datasets to official nnUNet v2 raw layout.")
    parser.add_argument("--dataset", choices=["tn3k", "kvasir", "isic"], required=True)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root path in this repo layout.")
    parser.add_argument("--nnunet-raw", type=Path, required=True, help="Path to nnUNet_raw directory.")
    parser.add_argument("--dataset-id", type=int, required=True, help="Three-digit ID used by nnUNet v2.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional custom dataset name suffix.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_name = args.dataset_name or args.dataset.upper()
    train_samples, test_samples = _resolve_samples(args.dataset, args.root)
    if len(train_samples) == 0:
        raise ValueError(f"No training samples found for dataset={args.dataset} root={args.root}")

    dataset_dir = _prepare_layout(args.nnunet_raw, args.dataset_id, dataset_name, args.overwrite)

    next_idx, max_ch = _copy_cases(
        train_samples,
        images_dir=dataset_dir / "imagesTr",
        labels_dir=dataset_dir / "labelsTr",
        prefix=f"{args.dataset}Tr",
        start_index=0,
        with_labels=True,
    )
    _, max_ch_ts = _copy_cases(
        test_samples,
        images_dir=dataset_dir / "imagesTs",
        labels_dir=None,
        prefix=f"{args.dataset}Ts",
        start_index=0,
        with_labels=False,
    )
    max_channels = max(max_ch, max_ch_ts)
    _write_dataset_json(dataset_dir, dataset_name, max_channels, num_training=next_idx)

    print(f"Prepared: {dataset_dir}")
    print(f"Training cases: {next_idx}")
    print(f"Test cases: {len(test_samples)}")
    print(f"Max channels: {max_channels}")


if __name__ == "__main__":
    main()
