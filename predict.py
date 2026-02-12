from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from data.segmentation import discover_drive_samples, discover_kvasir_samples, discover_synapse_volumes
from model.dinov3_decoder import DINOv3SegmentationModel, GuideDINOModel, SegDINOModel
from model.nnwnet import WNet2D
from model.swinunet import SwinUnet
from model.unet import GuideUNet, UNet


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedToken inference (nnUNet-style sliding window)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, default=None)
    parser.add_argument("--drive-split", type=str, default="training", choices=["training", "test"])
    parser.add_argument("--kvasir-root", type=Path, default=None)
    parser.add_argument("--synapse-root", type=Path, default=None)
    parser.add_argument("--synapse-to-rgb", action="store_true", default=False)
    parser.add_argument("--synapse-target-spacing", type=float, nargs=3, default=None)
    parser.add_argument("--synapse-crop-nonzero", action="store_true", default=True)
    parser.add_argument("--no-synapse-crop-nonzero", action="store_false", dest="synapse_crop_nonzero")
    parser.add_argument("--synapse-zscore", action="store_true", default=True)
    parser.add_argument("--no-synapse-zscore", action="store_false", dest="synapse_zscore")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, nargs="+", default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--mirror-tta", action="store_true", default=True)
    parser.add_argument("--no-mirror-tta", action="store_false", dest="mirror_tta")
    parser.add_argument("--seg-preprocess", choices=["nnunet", "dino", "dino_strong"], default="nnunet")
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument(
        "--model",
        choices=["swinunet", "unet", "guideunet", "dinov3", "segdino", "guidedino", "nnwnet"],
        default="swinunet",
    )
    parser.add_argument("--dinov3-backbone", type=str, default="facebook/dinov3-vit7b16-pretrain-lvd1689m")
    parser.add_argument("--dinov3-hidden-dim", type=int, default=256)
    parser.add_argument("--dinov3-dropout", type=float, default=0.0)
    parser.add_argument("--dinov3-train-backbone", action="store_true")
    parser.add_argument("--tokenbook-tokens", type=int, default=None)
    parser.add_argument("--tokenbook-dropout", type=float, default=0.0)
    parser.add_argument("--tokenbook-sample-rate", type=float, default=1.0)
    parser.add_argument("--tokenbook-ema-decay", type=float, default=None)
    parser.add_argument("--tokenbook-use-ema", action="store_true", default=False)
    parser.add_argument("--segdino-encoder-size", choices=["small", "base", "large", "giant"], default="base")
    parser.add_argument("--segdino-features", type=int, default=128)
    parser.add_argument("--segdino-out-channels", type=int, nargs=4, default=[96, 192, 384, 768])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _resolve_patch_size(patch_size: List[int] | None) -> Tuple[int, int] | None:
    if patch_size is None:
        return None
    if len(patch_size) == 1:
        return int(patch_size[0]), int(patch_size[0])
    if len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    raise ValueError("patch_size must be a single int or two ints")


def _zscore_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 2:
        mean = float(image.mean())
        std = float(image.std())
        return (image - mean) / (std + 1e-8)
    mean = image.mean(axis=(0, 1), keepdims=True)
    std = image.std(axis=(0, 1), keepdims=True)
    return (image - mean) / (std + 1e-8)


def _normalize_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "nnunet":
        return _zscore_image(image)
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def _gaussian_weight_map(patch_h: int, patch_w: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, patch_h, device=device),
        torch.linspace(-1.0, 1.0, patch_w, device=device),
        indexing="ij",
    )
    sigma = 0.5
    weight = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return weight


def _apply_mirror(model: torch.nn.Module, patch: torch.Tensor, mirror: bool) -> torch.Tensor:
    if not mirror:
        return model(patch)
    flips = [(False, False), (True, False), (False, True), (True, True)]
    outputs = []
    for flip_y, flip_x in flips:
        patch_aug = patch
        if flip_y:
            patch_aug = torch.flip(patch_aug, dims=[2])
        if flip_x:
            patch_aug = torch.flip(patch_aug, dims=[3])
        pred = model(patch_aug)
        if flip_y:
            pred = torch.flip(pred, dims=[2])
        if flip_x:
            pred = torch.flip(pred, dims=[3])
        outputs.append(pred)
    return torch.stack(outputs, dim=0).mean(dim=0)


def _sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Tuple[int, int],
    overlap: float,
    mirror: bool,
) -> torch.Tensor:
    _, _, height, width = image.shape
    patch_h, patch_w = patch_size
    stride_h = max(int(patch_h * (1 - overlap)), 1)
    stride_w = max(int(patch_w * (1 - overlap)), 1)

    pad_h = max(patch_h - height, 0)
    pad_w = max(patch_w - width, 0)
    if pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
        height = image.shape[2]
        width = image.shape[3]

    device = image.device
    weight_map = _gaussian_weight_map(patch_h, patch_w, device=device)
    output = None
    weight_sum = torch.zeros((1, 1, height, width), device=device)

    y_positions = list(range(0, height - patch_h + 1, stride_h))
    x_positions = list(range(0, width - patch_w + 1, stride_w))
    if y_positions[-1] != height - patch_h:
        y_positions.append(height - patch_h)
    if x_positions[-1] != width - patch_w:
        x_positions.append(width - patch_w)

    for y in y_positions:
        for x in x_positions:
            patch = image[:, :, y : y + patch_h, x : x + patch_w]
            pred = _apply_mirror(model, patch, mirror)
            if output is None:
                output = torch.zeros((1, pred.shape[1], height, width), device=device)
            output[:, :, y : y + patch_h, x : x + patch_w] += pred * weight_map
            weight_sum[:, :, y : y + patch_h, x : x + patch_w] += weight_map

    output = output / weight_sum.clamp_min(1e-8)
    return output[:, :, : image.shape[2] - pad_h, : image.shape[3] - pad_w]


def _save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    Image.fromarray(mask).save(output_path)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    filtered = {k.replace("model.", ""): v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "dinov3":
        model = DINOv3SegmentationModel(
            backbone_name=args.dinov3_backbone,
            train_backbone=args.dinov3_train_backbone,
            num_classes=args.num_classes,
            hidden_dim=args.dinov3_hidden_dim,
            dropout=args.dinov3_dropout,
        )
    elif args.model == "guidedino":
        model = GuideDINOModel(
            backbone_name=args.dinov3_backbone,
            train_backbone=args.dinov3_train_backbone,
            num_classes=args.num_classes,
            hidden_dim=args.dinov3_hidden_dim,
            dropout=args.dinov3_dropout,
            tokenbook_tokens=args.tokenbook_tokens,
            tokenbook_image_size=args.image_size,
            tokenbook_dropout=args.tokenbook_dropout,
            tokenbook_sample_rate=args.tokenbook_sample_rate,
            tokenbook_ema_decay=args.tokenbook_ema_decay,
            tokenbook_use_ema=args.tokenbook_use_ema,
        )
    elif args.model == "segdino":
        model = SegDINOModel(
            backbone_name=args.dinov3_backbone,
            train_backbone=args.dinov3_train_backbone,
            num_classes=args.num_classes,
            encoder_size=args.segdino_encoder_size,
            features=args.segdino_features,
            out_channels=args.segdino_out_channels,
        )
    elif args.model == "nnwnet":
        model = WNet2D(
            in_channel=args.in_chans,
            num_classes=args.num_classes,
            deep_supervised=False,
        )
    elif args.model == "unet":
        model = UNet(in_channels=args.in_chans, num_classes=args.num_classes)
    elif args.model == "guideunet":
        model = GuideUNet(
            in_channels=args.in_chans,
            num_classes=args.num_classes,
            guide_backbone_name=args.dinov3_backbone,
            guide_backbone_train=args.dinov3_train_backbone,
            tokenbook_tokens=args.tokenbook_tokens,
            tokenbook_image_size=args.image_size,
            tokenbook_dropout=args.tokenbook_dropout,
            tokenbook_sample_rate=args.tokenbook_sample_rate,
            tokenbook_ema_decay=args.tokenbook_ema_decay,
            tokenbook_use_ema=args.tokenbook_use_ema,
        )
    else:
        model = SwinUnet(
            img_size=args.image_size,
            patch_size=4,
            in_chans=args.in_chans,
            num_classes=args.num_classes,
        )
    return model


def _predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int] | None,
    overlap: float,
    mirror: bool,
    threshold: float,
) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[..., None]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    if patch_size is None:
        logits = _apply_mirror(model, image_tensor, mirror)
    else:
        logits = _sliding_window_predict(model, image_tensor, patch_size, overlap, mirror)

    if logits.shape[1] > 1:
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    else:
        prob = torch.sigmoid(logits).squeeze(0).squeeze(0)
        pred = (prob > threshold).cpu().numpy().astype(np.uint8) * 255
    return pred


def _iter_image_paths(args: argparse.Namespace) -> Iterable[Tuple[Path, str]]:
    if args.drive_root is not None:
        if args.drive_split == "test":
            image_dir = args.drive_root / "test" / "images"
            for image_path in sorted(image_dir.glob("*")):
                if image_path.is_file():
                    yield image_path, "drive"
        else:
            samples = discover_drive_samples(args.drive_root, split=args.drive_split, use_manual=True)
            for sample in samples:
                yield sample.image_path, "drive"
    if args.kvasir_root is not None:
        samples = discover_kvasir_samples(args.kvasir_root)
        for sample in samples:
            yield sample.image_path, "kvasir"


def _infer_synapse(
    model: torch.nn.Module,
    args: argparse.Namespace,
    patch_size: Tuple[int, int] | None,
) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Synapse inference requires nibabel. Install it with `pip install nibabel`.") from exc
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise ImportError("Synapse inference requires scipy. Install it with `pip install scipy`.") from exc

    volumes = discover_synapse_volumes(args.synapse_root)
    for volume in volumes:
        image_obj = nib.load(str(volume.image_path))
        image_vol = image_obj.get_fdata().astype(np.float32)

        if args.synapse_crop_nonzero:
            nonzero = image_vol != 0
            if np.any(nonzero):
                coords = np.array(np.where(nonzero))
                start = coords.min(axis=1)
                end = coords.max(axis=1) + 1
                slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
                image_vol = image_vol[slices]

        if args.synapse_zscore:
            mask = image_vol != 0
            if np.any(mask):
                mean = float(image_vol[mask].mean())
                std = float(image_vol[mask].std())
            else:
                mean = float(image_vol.mean())
                std = float(image_vol.std())
            image_vol = (image_vol - mean) / (std + 1e-8)

        if args.synapse_target_spacing is not None:
            current_spacing = image_obj.header.get_zooms()[: image_vol.ndim]
            zoom_factors = [cs / ts for cs, ts in zip(current_spacing, args.synapse_target_spacing)]
            image_vol = zoom(image_vol, zoom=zoom_factors, order=1)

        output_root = args.output_dir or volume.image_path.parent
        out_dir = output_root / f"{volume.image_path.stem}_pred"

        for slice_idx in range(image_vol.shape[-1]):
            image = image_vol.take(slice_idx, axis=image_vol.ndim - 1)
            if args.synapse_to_rgb:
                image = np.repeat(image[..., None], 3, axis=2)
            if args.seg_preprocess != "nnunet":
                image = _normalize_image(image, args.seg_preprocess)
            pred = _predict_image(model, image, patch_size, args.overlap, args.mirror_tta, args.threshold)
            out_path = out_dir / f"slice_{slice_idx:04d}.png"
            _save_mask(pred, out_path)


def main() -> None:
    args = _parse_args()
    patch_size = _resolve_patch_size(args.patch_size)

    model = _build_model(args)
    _load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if args.synapse_root is not None:
        _infer_synapse(model, args, patch_size)

    for image_path, source in _iter_image_paths(args):
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        image = _normalize_image(image, args.seg_preprocess)
        pred = _predict_image(model, image, patch_size, args.overlap, args.mirror_tta, args.threshold)
        output_root = args.output_dir or image_path.parent
        out_path = output_root / f"{image_path.stem}_pred.png"
        _save_mask(pred, out_path)
        print(f"Saved {source} prediction: {out_path}")


if __name__ == "__main__":
    main()
