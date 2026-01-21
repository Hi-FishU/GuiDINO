import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from data.segmentation import MedTokenSegmentationDataModule
from model.swinunet import SwinUnet
from model.wrapper import MedTokenSegLightningModule
from utils.loss import DC_and_BCE_loss, DC_and_CE_loss, DC_and_topk_loss

torch.set_float32_matmul_precision('high')

def build_criterion(args) -> torch.nn.Module:
    soft_dice_kwargs = {
        "batch_dice": False,
        "do_bg": True,
        "smooth": 1.0,
        "ddp": False,
    }

    if args.loss == "dc_bce":
        bce_kwargs = {}
        return DC_and_BCE_loss(
            bce_kwargs=bce_kwargs,
            soft_dice_kwargs=soft_dice_kwargs,
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

    raise ValueError(f"Unknown loss type: {args.loss}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedToken segmentation training")
    parser.add_argument("--drive-root", type=Path, default=None)
    parser.add_argument("--kvasir-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--drive-val-split", type=float, default=0.2)
    parser.add_argument("--kvasir-val-split", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=3e-5)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--no-nesterov", action="store_false", dest="nesterov", default=True)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight-ce", type=float, default=1.0)
    parser.add_argument("--weight-dice", type=float, default=1.0)
    parser.add_argument("--topk-percent", type=float, default=10.0)
    parser.add_argument("--loss", choices=["dc_bce", "dc_ce", "dc_topk"], default="dc_bce")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-chans", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--run-name", type=str, default="medtoken-seg")
    parser.add_argument("--run-model", type=str, default="swinunet")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    project = args.run_name
    name = args.run_model

    model = SwinUnet(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
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
    )

    data_module = MedTokenSegmentationDataModule(
        drive_root=args.drive_root,
        kvasir_root=args.kvasir_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        drive_val_split=args.drive_val_split,
        kvasir_val_split=args.kvasir_val_split,
    )

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=3),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    csv_logger = CSVLogger(save_dir=str(args.log_dir), name=args.run_name)
    wandb_logger = WandbLogger(project=project, name=name, log_model=True)

    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=[csv_logger, wandb_logger],
        log_every_n_steps=10,
    )

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
