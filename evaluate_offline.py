from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from data.segmentation import (
    discover_drive_samples,
    discover_isic_samples,
    discover_kvasir_samples,
    discover_tn3k_samples,
)


def _build_case_pairs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if args.kvasir_root is not None:
        samples = discover_kvasir_samples(args.kvasir_root)
        for s in samples:
            pred = args.pred_dir / f"{s.image_path.stem}{args.pred_suffix}"
            if pred.exists():
                pairs.append((pred, s.mask_path))
    if args.drive_root is not None:
        samples = discover_drive_samples(args.drive_root, split=args.drive_split, use_manual=True)
        for s in samples:
            pred = args.pred_dir / f"{s.image_path.stem}{args.pred_suffix}"
            if pred.exists():
                pairs.append((pred, s.mask_path))
    if args.isic_root is not None:
        samples = discover_isic_samples(args.isic_root)
        for s in samples:
            pred = args.pred_dir / f"{s.image_path.stem}{args.pred_suffix}"
            if pred.exists():
                pairs.append((pred, s.mask_path))
    if args.tn3k_root is not None:
        samples = discover_tn3k_samples(args.tn3k_root, split=args.tn3k_split)
        for s in samples:
            pred = args.pred_dir / f"{s.image_path.stem}{args.pred_suffix}"
            if pred.exists():
                pairs.append((pred, s.mask_path))
    if not pairs:
        raise RuntimeError("No prediction/GT pairs found. Check --pred-dir and dataset root.")
    return pairs


def _load_binary(path: Path, threshold: float) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    if arr.dtype == np.uint8:
        thr = threshold * 255.0 if threshold <= 1.0 else threshold
    else:
        thr = threshold
    return (arr > thr).astype(np.uint8)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask.astype(bool))
    if n <= 1:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(counts.argmax())
    return (labeled == keep).astype(np.uint8)


def _dice_iou(mask_pred: np.ndarray, mask_ref: np.ndarray) -> Tuple[float, float]:
    pred = mask_pred.astype(bool)
    ref = mask_ref.astype(bool)
    tp = np.sum(pred & ref)
    fp = np.sum(pred & (~ref))
    fn = np.sum((~pred) & ref)
    if tp + fp + fn == 0:
        return float("nan"), float("nan")
    dice = 2.0 * tp / (2.0 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    return float(dice), float(iou)


def _surface_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Optional[Tuple[float, float]],
    empty_policy: str,
) -> Tuple[float, float]:
    pred_bool = pred.astype(bool)
    ref_bool = ref.astype(bool)

    if pred_bool.sum() == 0 and ref_bool.sum() == 0:
        return float("nan"), float("nan")

    if pred_bool.sum() == 0 or ref_bool.sum() == 0:
        if empty_policy == "ignore":
            return float("nan"), float("nan")
        if spacing is None:
            diag = float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))
        else:
            diag = float(
                np.sqrt(
                    (pred.shape[0] * float(spacing[0])) ** 2
                    + (pred.shape[1] * float(spacing[1])) ** 2
                )
            )
        return diag, diag

    pred_border = pred_bool ^ binary_erosion(pred_bool)
    ref_border = ref_bool ^ binary_erosion(ref_bool)
    dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
    dt_ref = distance_transform_edt(~ref_bool, sampling=spacing)

    surface_distances: List[np.ndarray] = []
    d_ref_to_pred = dt_pred[ref_border]
    if d_ref_to_pred.size:
        surface_distances.append(d_ref_to_pred)
    d_pred_to_ref = dt_ref[pred_border]
    if d_pred_to_ref.size:
        surface_distances.append(d_pred_to_ref)
    if not surface_distances:
        return 0.0, 0.0

    all_d = np.concatenate(surface_distances)
    return float(np.percentile(all_d, 95)), float(np.mean(all_d))


def _summarize(values: Sequence[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _evaluate_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    threshold: float,
    keep_largest: bool,
    spacing: Optional[Tuple[float, float]],
    empty_policy: str,
    surface_aggregation: str,
) -> Dict:
    dice_values: List[float] = []
    iou_values: List[float] = []
    hd95_values: List[float] = []
    asd_values: List[float] = []
    hd95_nonempty: List[float] = []
    asd_nonempty: List[float] = []

    per_case: List[Dict] = []
    for pred_path, gt_path in pairs:
        pred = _load_binary(pred_path, threshold)
        gt = _load_binary(gt_path, threshold=0.5)
        if keep_largest:
            pred = _keep_largest_component(pred)

        dice, iou = _dice_iou(pred, gt)
        hd95, asd = _surface_metrics(pred, gt, spacing=spacing, empty_policy=empty_policy)

        dice_values.append(dice)
        iou_values.append(iou)
        hd95_values.append(hd95)
        asd_values.append(asd)

        if pred.astype(bool).sum() > 0 and gt.astype(bool).sum() > 0:
            hd95_nonempty.append(hd95)
            asd_nonempty.append(asd)

        per_case.append(
            {
                "prediction_file": str(pred_path),
                "reference_file": str(gt_path),
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "asd": asd,
                "pred_fg": int(pred.astype(bool).sum()),
                "ref_fg": int(gt.astype(bool).sum()),
            }
        )

    if surface_aggregation == "nonempty":
        hd95_mean = _summarize(hd95_nonempty)
        asd_mean = _summarize(asd_nonempty)
    else:
        hd95_mean = _summarize(hd95_values)
        asd_mean = _summarize(asd_values)

    return {
        "num_cases": len(pairs),
        "mean": {
            "dice": _summarize(dice_values),
            "iou": _summarize(iou_values),
            "hd95": hd95_mean,
            "asd": asd_mean,
            "hd95_all": _summarize(hd95_values),
            "asd_all": _summarize(asd_values),
            "hd95_nonempty": _summarize(hd95_nonempty),
            "asd_nonempty": _summarize(asd_nonempty),
            "surface_nonempty_count": len(hd95_nonempty),
        },
        "per_case": per_case,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline segmentation evaluation with nnUNet-style options.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing predicted masks.")
    parser.add_argument("--pred-suffix", type=str, default="_pred.png", help="Pred filename suffix.")
    parser.add_argument("--kvasir-root", type=Path, default=None)
    parser.add_argument("--isic-root", type=Path, default=None)
    parser.add_argument("--tn3k-root", type=Path, default=None)
    parser.add_argument("--tn3k-split", type=str, default="test", choices=["trainval", "test"])
    parser.add_argument("--drive-root", type=Path, default=None)
    parser.add_argument("--drive-split", type=str, default="training", choices=["training", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-postprocess-largest", action="store_true", default=False)
    parser.add_argument("--keep-largest-component", action="store_true", default=False)
    parser.add_argument("--surface-metric-spacing", type=float, nargs=2, default=None)
    parser.add_argument("--surface-empty-policy", choices=["penalize", "ignore"], default="penalize")
    parser.add_argument("--surface-aggregation", choices=["nonempty", "all"], default="nonempty")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kvasir_root is None and args.drive_root is None and args.isic_root is None and args.tn3k_root is None:
        raise ValueError(
            "Provide at least one dataset root: --kvasir-root, --isic-root, --tn3k-root, and/or --drive-root"
        )
    pairs = _build_case_pairs(args)

    spacing = tuple(args.surface_metric_spacing) if args.surface_metric_spacing else None
    base_eval = _evaluate_pairs(
        pairs,
        threshold=args.threshold,
        keep_largest=args.keep_largest_component,
        spacing=spacing,
        empty_policy=args.surface_empty_policy,
        surface_aggregation=args.surface_aggregation,
    )

    selected_keep_largest = bool(args.keep_largest_component)
    if args.auto_postprocess_largest:
        no_pp = _evaluate_pairs(
            pairs,
            threshold=args.threshold,
            keep_largest=False,
            spacing=spacing,
            empty_policy=args.surface_empty_policy,
            surface_aggregation=args.surface_aggregation,
        )
        with_pp = _evaluate_pairs(
            pairs,
            threshold=args.threshold,
            keep_largest=True,
            spacing=spacing,
            empty_policy=args.surface_empty_policy,
            surface_aggregation=args.surface_aggregation,
        )
        no_pp_dice = no_pp["mean"]["dice"]
        with_pp_dice = with_pp["mean"]["dice"]
        if np.isnan(no_pp_dice) or (not np.isnan(with_pp_dice) and with_pp_dice > no_pp_dice):
            base_eval = with_pp
            selected_keep_largest = True
        else:
            base_eval = no_pp
            selected_keep_largest = False

    result = {
        "settings": {
            "pred_dir": str(args.pred_dir),
            "pred_suffix": args.pred_suffix,
            "threshold": args.threshold,
            "keep_largest_component": selected_keep_largest,
            "auto_postprocess_largest": bool(args.auto_postprocess_largest),
            "surface_metric_spacing": list(spacing) if spacing is not None else None,
            "surface_empty_policy": args.surface_empty_policy,
            "surface_aggregation": args.surface_aggregation,
        },
        **base_eval,
    }

    output_json = args.output_json or (args.pred_dir / "summary_boundary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))

    mean = result["mean"]
    print(f"Cases: {result['num_cases']}")
    print(f"Dice: {mean['dice']:.6f} | IoU: {mean['iou']:.6f}")
    print(f"HD95: {mean['hd95']:.6f} | ASD: {mean['asd']:.6f}")
    print(
        f"(all) HD95: {mean['hd95_all']:.6f} | ASD: {mean['asd_all']:.6f} | "
        f"nonempty_count: {mean['surface_nonempty_count']}"
    )
    print(f"Saved summary: {output_json}")


if __name__ == "__main__":
    main()
