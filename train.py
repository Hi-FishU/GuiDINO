import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Disable Albumentations online version check warnings in offline environments.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from data.segmentation import MedTokenSegmentationDataModule
from model.swinunet import SwinUnet
from model.unet import UNet
from model.dinov3_decoder import DINOv3SegmentationModel, GuideDINOModel, SegDINOModel
from model.unet import GuideUNet
from model.nnwnet import GuideWNet2D, WNet2D
from model.wrapper import MedTokenSegLightningModule
from utils.loss import (
    DC_and_BCE_loss,
    DC_and_BCE_and_HingeD_loss,
    DC_and_CE_loss,
    DC_and_topk_loss,
    Guide_DC_and_BCE_and_HingeD_loss,
    Guide_DC_and_BCE_loss,
    Guide_DC_and_CE_loss,
)

torch.set_float32_matmul_precision('high')


class ChannelsLastWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images, *args, **kwargs):
        if torch.is_tensor(images):
            images = images.to(memory_format=torch.channels_last)
        return self.model(images, *args, **kwargs)


def _uses_imagenet_norm(seg_preprocess: str) -> bool:
    return seg_preprocess in {"dino", "dino_strong"}


def build_criterion(args) -> torch.nn.Module:
    soft_dice_kwargs = {
        "batch_dice": args.dice_batch_dice,
        "do_bg": args.dice_do_bg,
        "smooth": args.dice_smooth,
        "ddp": False,
    }
    # BCE-style binary segmentation is single-channel in this codebase; excluding
    # background would drop the only channel and produce NaNs in Dice.
    soft_dice_kwargs_bce = dict(soft_dice_kwargs, do_bg=True)

    if args.loss == "dc_bce":
        bce_kwargs = {}
        return DC_and_BCE_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs_bce,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            use_ignore_label=False,
        )

    if args.loss == "dc_ce":
        ce_kwargs = {}
        return DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            ignore_label=None,
        )

    if args.loss == "dc_topk":
        ce_kwargs = {"k": args.topk_percent}
        return DC_and_topk_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            ignore_label=None,
        )
    if args.loss == "guide_dc_bce":
        bce_kwargs = {}
        return Guide_DC_and_BCE_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs_bce,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            use_ignore_label=False,
            weight_guide=args.weight_guide,
        )
    if args.loss == "dc_bce_hinged":
        bce_kwargs = {}
        return DC_and_BCE_and_HingeD_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs_bce,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            use_ignore_label=False,
            weight_hinge_d=args.weight_hinge_d,
            hinge_d_margin=args.hinge_d_margin,
            hinge_d_kernel_size=args.hinge_d_kernel_size,
            hinge_d_boundary_only=args.hinge_d_boundary_only,
        )
    if args.loss == "guide_dc_ce":
        ce_kwargs = {}
        return Guide_DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            ignore_label=None,
            weight_guide=args.weight_guide,
        )
    if args.loss == "guide_dc_bce_hinged":
        bce_kwargs = {}
        return Guide_DC_and_BCE_and_HingeD_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs_bce,
            weight_ce=args.weight_ce,
            weight_dice=args.weight_dice,
            use_ignore_label=False,
            weight_guide=args.weight_guide,
            weight_hinge_d=args.weight_hinge_d,
            hinge_d_margin=args.hinge_d_margin,
            hinge_d_kernel_size=args.hinge_d_kernel_size,
            hinge_d_boundary_only=args.hinge_d_boundary_only,
        )

    raise ValueError(f"Unknown loss type: {args.loss}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedToken segmentation training")
    parser.add_argument("--drive-root", type=Path, default=None)
    parser.add_argument("--kvasir-root", type=Path, default=None)
    parser.add_argument("--synapse-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--patch-train-size", type=int, nargs="+", default=None)
    parser.add_argument("--val-sw-patch-size", type=int, nargs="+", default=None)
    parser.add_argument("--oversample-foreground", type=float, default=0.33)
    parser.add_argument("--fullres-val-eval", action="store_true", default=True)
    parser.add_argument("--no-fullres-val-eval", action="store_false", dest="fullres_val_eval")
    parser.add_argument("--val-sw-overlap", type=float, default=0.5)
    parser.add_argument("--val-sw-mirror", action="store_true", default=True)
    parser.add_argument("--no-val-sw-mirror", action="store_false", dest="val_sw_mirror")
    parser.add_argument("--val-sw-mirror-axes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--val-sw-gaussian-sigma-scale", type=float, default=1.0 / 8.0)
    parser.add_argument("--val-sw-gaussian-value-scaling", type=float, default=10.0)
    parser.add_argument("--drive-val-split", type=float, default=0.2)
    parser.add_argument("--kvasir-val-split", type=float, default=0.1)
    parser.add_argument("--synapse-val-split", type=float, default=0.2)
    parser.add_argument("--synapse-include-empty", action="store_true", default=False)
    parser.add_argument("--synapse-to-rgb", action="store_true", default=False)
    parser.add_argument("--synapse-cache", action="store_true", default=False)
    parser.add_argument("--synapse-target-spacing", type=float, nargs=3, default=None)
    parser.add_argument("--synapse-crop-nonzero", action="store_true", default=True)
    parser.add_argument("--no-synapse-crop-nonzero", action="store_false", dest="synapse_crop_nonzero")
    parser.add_argument("--synapse-zscore", action="store_true", default=True)
    parser.add_argument("--no-synapse-zscore", action="store_false", dest="synapse_zscore")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--no-nesterov", action="store_false", dest="nesterov", default=True)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--adamw-betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument(
        "--lr-scheduler",
        choices=["poly", "cosine", "cosine_restart", "none"],
        default="poly",
    )
    parser.add_argument("--cosine-t-max", type=int, default=None)
    parser.add_argument("--cosine-restart-t-0", type=int, default=10)
    parser.add_argument("--cosine-restart-t-mult", type=int, default=2)
    parser.add_argument("--cosine-restart-eta-min", type=float, default=0.0)
    parser.add_argument("--best-metric", type=str, default="val/dice")
    parser.add_argument("--best-metric-mode", choices=["max", "min"], default="max")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-keep-largest-component", action="store_true", default=False)
    parser.add_argument("--surface-metric-spacing", type=float, nargs=2, default=None)
    parser.add_argument("--weight-ce", type=float, default=1.0)
    parser.add_argument("--weight-dice", type=float, default=1.0)
    parser.add_argument("--weight-guide", type=float, default=0.1)
    parser.add_argument("--weight-hinge-d", type=float, default=0.1)
    parser.add_argument("--dice-do-bg", action="store_true", default=False)
    parser.add_argument("--no-dice-do-bg", action="store_false", dest="dice_do_bg")
    parser.add_argument("--dice-batch-dice", action="store_true", default=True)
    parser.add_argument("--no-dice-batch-dice", action="store_false", dest="dice_batch_dice")
    parser.add_argument("--dice-smooth", type=float, default=1e-5)
    parser.add_argument("--hinge-d-margin", type=float, default=1.0)
    parser.add_argument("--hinge-d-kernel-size", type=int, default=3)
    parser.add_argument("--hinge-d-boundary-only", action="store_true", default=True)
    parser.add_argument("--no-hinge-d-boundary-only", action="store_false", dest="hinge_d_boundary_only")
    parser.add_argument("--topk-percent", type=float, default=10.0)
    parser.add_argument(
        "--loss",
        choices=[
            "dc_bce",
            "dc_ce",
            "dc_topk",
            "dc_bce_hinged",
            "guide_dc_ce",
            "guide_dc_bce",
            "guide_dc_bce_hinged",
        ],
        default="dc_bce",
    )
    parser.add_argument(
        "--seg-preprocess",
        choices=["nnunet", "dino", "dino_strong"],
        default="nnunet",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--run-name", type=str, default="medtoken-seg")
    parser.add_argument("--run-model", type=str, default="smoke-test")
    parser.add_argument(
        "--model",
        choices=["swinunet", "unet", "guideunet", "dinov3", "segdino", "guidedino", "nnwnet", "guidennwnet"],
        default="swinunet",
        help="Backbone/decoder choice for segmentation.",
    )
    parser.add_argument("--nnwnet-deep-supervision", action="store_true", default=True)
    parser.add_argument("--no-nnwnet-deep-supervision", action="store_false", dest="nnwnet_deep_supervision")
    parser.add_argument(
        "--use-guide",
        action="store_true",
        default=False,
        help="Shortcut to enable --model guidedino and --loss guide_dc_bce together.",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", action="store_false", dest="tf32")
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", action="store_false", dest="channels_last")
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark")
    parser.add_argument("--train-epoch-eval", action="store_true", default=True)
    parser.add_argument("--no-train-epoch-eval", action="store_false", dest="train_epoch_eval")
    parser.add_argument("--log-image-samples", type=int, default=0)
    parser.add_argument("--log-image-every-n-epochs", type=int, default=1)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--compile-dynamic", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--dinov3-backbone",
        type=str,
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    )
    parser.add_argument("--dinov3-hidden-dim", type=int, default=256)
    parser.add_argument("--dinov3-dropout", type=float, default=0.0)
    parser.add_argument("--dinov3-train-backbone", action="store_true")
    parser.add_argument("--tokenbook-tokens", type=int, default=None)
    parser.add_argument("--tokenbook-dropout", type=float, default=0.0)
    parser.add_argument("--tokenbook-sample-rate", type=float, default=1.0)
    parser.add_argument("--tokenbook-ema-decay", type=float, default=None)
    parser.add_argument("--tokenbook-use-ema", action="store_true", default=False)
    parser.add_argument(
        "--segdino-encoder-size",
        choices=["small", "base", "large", "giant"],
        default="base",
    )
    parser.add_argument("--segdino-features", type=int, default=128)
    parser.add_argument(
        "--segdino-out-channels",
        type=int,
        nargs=4,
        default=[96, 192, 384, 768],
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(axis not in (0, 1) for axis in args.val_sw_mirror_axes):
        raise ValueError(
            f"Invalid --val-sw-mirror-axes {args.val_sw_mirror_axes}. For 2D use only 0 (H) and/or 1 (W)."
        )
    if not (0.0 <= args.val_sw_overlap < 1.0):
        raise ValueError("--val-sw-overlap must be in [0.0, 1.0).")
    if args.use_guide:
        args.model = "guidedino"
        if args.loss in ("dc_bce", "dc_bce_hinged"):
            args.loss = "guide_dc_bce"
    if args.synapse_root is not None:
        if args.num_classes == 1:
            args.num_classes = 9
        dino_models = {"dinov3", "segdino", "guidedino", "guideunet", "guidennwnet"}
        if args.model in dino_models:
            if args.in_chans == 1:
                args.in_chans = 3
            if not args.synapse_to_rgb:
                args.synapse_to_rgb = True
        else:
            if args.in_chans == 3:
                args.in_chans = 1
        if _uses_imagenet_norm(args.seg_preprocess) and args.synapse_zscore:
            args.synapse_zscore = False
            print(
                "Info: Disabled --synapse-zscore because --seg-preprocess "
                f"is '{args.seg_preprocess}' (DINO/ImageNet normalization is preserved)."
            )
        if args.loss in ("dc_bce", "dc_bce_hinged", "guide_dc_bce", "guide_dc_bce_hinged"):
            raise ValueError("Synapse is multi-class; use --loss dc_ce or dc_topk with num_classes > 1.")
    pl.seed_everything(args.seed, workers=True)
    project = args.run_name
    name = args.run_model

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # Default segmentation model is SwinUnet; switch to DINOv3 with --model dinov3/segdino/guidedino.
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
            deep_supervised=args.nnwnet_deep_supervision,
        )
    elif args.model == "guidennwnet":
        model = GuideWNet2D(
            in_channel=args.in_chans,
            num_classes=args.num_classes,
            deep_supervised=args.nnwnet_deep_supervision,
            guide_backbone_name=args.dinov3_backbone,
            guide_backbone_train=args.dinov3_train_backbone,
            tokenbook_tokens=args.tokenbook_tokens,
            tokenbook_image_size=args.image_size,
            tokenbook_dropout=args.tokenbook_dropout,
            tokenbook_sample_rate=args.tokenbook_sample_rate,
            tokenbook_ema_decay=args.tokenbook_ema_decay,
            tokenbook_use_ema=args.tokenbook_use_ema,
        )
    elif args.model == "unet":
        model = UNet(
            in_channels=args.in_chans,
            num_classes=args.num_classes,
        )
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
            patch_size=args.patch_size,
            in_chans=args.in_chans,
            num_classes=args.num_classes,
        )
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        model = ChannelsLastWrapper(model)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")
        model = torch.compile(
            model,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
        )
    guide_losses = {"guide_dc_ce", "guide_dc_bce", "guide_dc_bce_hinged"}
    if (args.model in ("guidedino", "guideunet", "guidennwnet")) != (args.loss in guide_losses):
        raise ValueError(
            "guidedino/guideunet/guidennwnet model requires a guide loss "
            "(guide_dc_bce or guide_dc_bce_hinged), and guide losses "
            "require --model guidedino, guideunet, or guidennwnet."
        )
    criterion = build_criterion(args)

    lightning_module = MedTokenSegLightningModule(
        model=model,
        criterion=criterion,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.nesterov,
        threshold=args.threshold,
        max_epochs=args.max_epochs,
        poly_power=args.poly_power,
        optimizer_name=args.optimizer,
        adamw_betas=tuple(args.adamw_betas),
        adamw_eps=args.adamw_eps,
        scheduler_name=args.lr_scheduler,
        cosine_t_max=args.cosine_t_max,
        cosine_restart_t_0=args.cosine_restart_t_0,
        cosine_restart_t_mult=args.cosine_restart_t_mult,
        cosine_restart_eta_min=args.cosine_restart_eta_min,
        best_metric_name=args.best_metric,
        best_metric_mode=args.best_metric_mode,
        train_epoch_eval=args.train_epoch_eval,
        log_image_samples=args.log_image_samples,
        log_image_every_n_epochs=args.log_image_every_n_epochs,
        val_use_sliding_window=args.fullres_val_eval,
        val_sw_patch_size=tuple(args.val_sw_patch_size) if args.val_sw_patch_size else (
            tuple(args.patch_train_size) if args.patch_train_size else (args.image_size, args.image_size)
        ),
        val_sw_overlap=args.val_sw_overlap,
        val_sw_mirror=args.val_sw_mirror,
        val_sw_mirror_axes=tuple(sorted(set(args.val_sw_mirror_axes))),
        val_sw_gaussian_sigma_scale=args.val_sw_gaussian_sigma_scale,
        val_sw_gaussian_value_scaling=args.val_sw_gaussian_value_scaling,
        eval_keep_largest_component=args.eval_keep_largest_component,
        surface_metric_spacing=tuple(args.surface_metric_spacing) if args.surface_metric_spacing else None,
    )

    data_module = MedTokenSegmentationDataModule(
        drive_root=args.drive_root,
        kvasir_root=args.kvasir_root,
        synapse_root=args.synapse_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        drive_val_split=args.drive_val_split,
        kvasir_val_split=args.kvasir_val_split,
        synapse_val_split=args.synapse_val_split,
        synapse_include_empty=args.synapse_include_empty,
        synapse_to_rgb=args.synapse_to_rgb,
        synapse_cache=args.synapse_cache,
        synapse_target_spacing=tuple(args.synapse_target_spacing) if args.synapse_target_spacing else None,
        synapse_crop_nonzero=args.synapse_crop_nonzero,
        synapse_zscore=args.synapse_zscore,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
        preprocessing=args.seg_preprocess,
        patch_size=tuple(args.patch_train_size) if args.patch_train_size else None,
        oversample_foreground_prob=args.oversample_foreground,
        full_res_val_eval=args.fullres_val_eval,
    )

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    csv_logger = CSVLogger(save_dir=str(args.log_dir), name=args.run_name)
    wandb_logger = WandbLogger(project=project, name=name, log_model=True)

    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    if accelerator == "gpu" and args.amp:
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    else:
        precision = "32-true"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=[csv_logger, wandb_logger],
        log_every_n_steps=4,
        precision=precision,
    )

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
