#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import sys
import torch
from torch.utils.data import DataLoader

# Allow running this script directly from anywhere, e.g.:
# python tools/benchmark_loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    get_segmentation_train_transform,
    get_segmentation_train_transform_dino,
    get_segmentation_train_transform_dino_strong,
)
from data.segmentation import (
    GenericSegmentationDataset,
    SegmentationSample,
    discover_isic_samples,
    discover_kvasir_samples,
)


@dataclass
class BenchResult:
    name: str
    mode: str
    num_samples: int
    seconds: float

    @property
    def samples_per_sec(self) -> float:
        if self.seconds <= 0:
            return float("inf")
        return self.num_samples / self.seconds


def _resolve_transform(name: str, image_size: int):
    if name == "nnunet":
        return get_segmentation_train_transform(image_size)
    if name == "dino":
        return get_segmentation_train_transform_dino(image_size)
    if name == "dino_strong":
        return get_segmentation_train_transform_dino_strong(image_size)
    raise ValueError(f"Unsupported preprocess: {name}")


def _slice_samples(samples: Sequence[SegmentationSample], max_samples: int) -> list[SegmentationSample]:
    if max_samples <= 0:
        return list(samples)
    return list(samples[: max_samples])


def _bench_getitem(name: str, dataset: GenericSegmentationDataset) -> BenchResult:
    n = len(dataset)
    t0 = time.perf_counter()
    for i in range(n):
        _ = dataset[i]
    dt = time.perf_counter() - t0
    return BenchResult(name=name, mode="getitem", num_samples=n, seconds=dt)


def _bench_dataloader(
    name: str,
    dataset: GenericSegmentationDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    mp_context: Optional[str],
) -> BenchResult:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["multiprocessing_context"] = mp_context
    loader = DataLoader(**kwargs)
    total = 0
    t0 = time.perf_counter()
    for images, _masks in loader:
        total += int(images.shape[0])
    dt = time.perf_counter() - t0
    return BenchResult(name=name, mode="dataloader", num_samples=total, seconds=dt)


def _load_samples(dataset_name: str, root: Path) -> list[SegmentationSample]:
    if dataset_name == "kvasir":
        return discover_kvasir_samples(root)
    if dataset_name == "isic":
        return discover_isic_samples(root)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _run_one(
    dataset_name: str,
    root: Path,
    transform_factory: Callable[[], object],
    max_samples: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    mp_context: Optional[str],
) -> list[BenchResult]:
    samples = _load_samples(dataset_name, root)
    if not samples:
        raise RuntimeError(f"No samples discovered for {dataset_name} at {root}")
    samples = _slice_samples(samples, max_samples=max_samples)
    transform = transform_factory()
    dataset = GenericSegmentationDataset(
        samples,
        transform=transform,
        patch_size=None,
        oversample_foreground_prob=0.0,
    )

    results = [_bench_getitem(dataset_name, dataset)]
    results.append(
        _bench_dataloader(
            dataset_name,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            mp_context=mp_context,
        )
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark segmentation dataset loading throughput.")
    parser.add_argument("--kvasir-root", type=Path, default=None)
    parser.add_argument("--isic-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--mp-context",
        choices=["default", "fork", "spawn", "forkserver"],
        default="default",
        help="Only used when --num-workers > 0.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["nnunet", "dino", "dino_strong"],
        default="dino_strong",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kvasir_root is None and args.isic_root is None:
        raise ValueError("Provide at least one root: --kvasir-root and/or --isic-root")

    mp_context = None if args.mp_context == "default" else args.mp_context
    transform_factory = lambda: _resolve_transform(args.preprocess, args.image_size)

    runs: list[tuple[str, Path]] = []
    if args.kvasir_root is not None:
        runs.append(("kvasir", args.kvasir_root))
    if args.isic_root is not None:
        runs.append(("isic", args.isic_root))

    print(
        f"settings preprocess={args.preprocess} image_size={args.image_size} "
        f"max_samples={args.max_samples} batch_size={args.batch_size} "
        f"num_workers={args.num_workers} mp_context={args.mp_context}"
    )
    print("-" * 80)

    all_results: list[BenchResult] = []
    for dataset_name, root in runs:
        results = _run_one(
            dataset_name=dataset_name,
            root=root,
            transform_factory=transform_factory,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            mp_context=mp_context,
        )
        all_results.extend(results)
        for r in results:
            print(
                f"{r.name:7s} {r.mode:10s} "
                f"samples={r.num_samples:4d} time={r.seconds:8.3f}s "
                f"throughput={r.samples_per_sec:8.2f} samples/s"
            )

    if args.kvasir_root is not None and args.isic_root is not None:
        print("-" * 80)
        by_key = {(r.name, r.mode): r for r in all_results}
        for mode in ("getitem", "dataloader"):
            k = by_key.get(("kvasir", mode))
            i = by_key.get(("isic", mode))
            if k is None or i is None:
                continue
            ratio = k.samples_per_sec / max(i.samples_per_sec, 1e-12)
            print(
                f"ratio {mode:10s}: Kvasir/ISIC throughput = {ratio:.2f}x "
                f"(higher means ISIC is slower)"
            )


if __name__ == "__main__":
    main()
