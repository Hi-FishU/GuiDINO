#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

# Allow running this script directly from anywhere, e.g.:
# python tools/build_segmentation_cache.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.segmentation import (
    SegmentationSample,
    discover_isic_samples,
    discover_kvasir_samples,
)


def _resize_keep_aspect(image: Image.Image, max_long_side: int, resample: int) -> Image.Image:
    w, h = image.size
    long_side = max(w, h)
    if max_long_side <= 0 or long_side <= max_long_side:
        return image
    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=resample)


def _load_samples(dataset: str, root: Path) -> list[SegmentationSample]:
    if dataset == "isic":
        return discover_isic_samples(root)
    if dataset == "kvasir":
        return discover_kvasir_samples(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cached segmentation dataset by pre-resizing images/masks."
    )
    parser.add_argument("--dataset", choices=["isic", "kvasir"], required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=1024,
        help="Resize preserving aspect ratio so the longer side is <= this value.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--image-quality", type=int, default=95, help="JPEG quality for cached images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _load_samples(args.dataset, args.src_root)
    if not samples:
        raise RuntimeError(f"No samples found for {args.dataset} under {args.src_root}")

    image_dir = args.dst_root / "images"
    mask_dir = args.dst_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building cache: dataset={args.dataset} samples={len(samples)} "
        f"max_long_side={args.max_long_side} dst={args.dst_root}"
    )

    for idx, sample in enumerate(samples, start=1):
        out_image = image_dir / f"{sample.name}.jpg"
        out_mask = mask_dir / f"{sample.name}.png"
        if not args.overwrite and out_image.exists() and out_mask.exists():
            continue

        with Image.open(sample.image_path) as img:
            img = img.convert("RGB")
            img = _resize_keep_aspect(img, args.max_long_side, Image.Resampling.LANCZOS)
            img.save(out_image, format="JPEG", quality=args.image_quality, optimize=True)

        with Image.open(sample.mask_path) as msk:
            msk = msk.convert("L")
            msk = _resize_keep_aspect(msk, args.max_long_side, Image.Resampling.NEAREST)
            msk.save(out_mask, format="PNG", optimize=True)

        if idx % 200 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] cached")

    print("Done.")


if __name__ == "__main__":
    main()
