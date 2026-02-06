
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader


class MedTokenSegLightningModule(pl.LightningModule):
    """
    Generic LightningModule wrapper for binary segmentation models.

    The wrapped `model` is expected to return either a tensor of logits or a mapping
    that contains segmentation logits (e.g. under `pred_masks`, `logits`, `masks`, etc.).
    The `criterion` should accept `(predictions, targets)` and return either a tensor
    loss or a dict of loss components.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Any,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        momentum: float = 0.99,
        nesterov: bool = True,
        threshold: float = 0.5,
        max_epochs: int = 1000,
        poly_power: float = 0.9,
        optimizer_name: str = "sgd",
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        scheduler_name: str = "poly",
        cosine_t_max: Optional[int] = None,
        cosine_restart_t_0: Optional[int] = None,
        cosine_restart_t_mult: int = 1,
        cosine_restart_eta_min: float = 0.0,
        best_metric_name: str = "val/dice",
        best_metric_mode: str = "max",
        train_epoch_eval: bool = True,
        log_image_samples: int = 0,
        log_image_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.threshold = threshold
        self.max_epochs = max_epochs
        self.poly_power = poly_power
        self.optimizer_name = optimizer_name
        self.adamw_betas = adamw_betas
        self.adamw_eps = adamw_eps
        self.scheduler_name = scheduler_name
        self.cosine_t_max = cosine_t_max
        self.cosine_restart_t_0 = cosine_restart_t_0
        self.cosine_restart_t_mult = cosine_restart_t_mult
        self.cosine_restart_eta_min = cosine_restart_eta_min
        self.best_metric_name = best_metric_name
        self.best_metric_mode = best_metric_mode
        self.train_epoch_eval = train_epoch_eval
        self.log_image_samples = int(log_image_samples)
        self.log_image_every_n_epochs = max(int(log_image_every_n_epochs), 1)
        self.best_metric_value: Optional[float] = None

    def forward(self, images: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if targets is not None:
            try:
                return self.model(images, targets)
            except TypeError:
                pass
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        loss_dict, loss = self._compute_loss(outputs, targets)
        for name, value in loss_dict.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=images.size(0),
            )
        # self.log(
        #     "train/loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=images.size(0),
        # )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        loss_dict, loss = self._compute_loss(outputs, targets)
        pred_masks = self._extract_pred_masks(outputs)
        metrics = self._compute_segmentation_metrics(pred_masks, targets)

        for name, value in loss_dict.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=images.size(0),
            )
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=metric_name in {"dice", "jaccard"},
                batch_size=images.size(0),
            )

        # self.log(
        #     "val/loss",
        #     loss,
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=images.size(0),
        # )
        if self._should_log_images(batch_idx):
            self._log_images(images, targets, outputs)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.best_metric_name is None:
            return
        metric = self.trainer.callback_metrics.get(self.best_metric_name)
        if metric is None:
            return
        if torch.is_tensor(metric):
            metric_value = metric.detach().float().item()
        else:
            metric_value = float(metric)
        if self.best_metric_value is None:
            improved = True
        elif self.best_metric_mode == "min":
            improved = metric_value < self.best_metric_value
        else:
            improved = metric_value > self.best_metric_value
        if improved:
            self.best_metric_value = metric_value
        if self.best_metric_value is not None:
            self.log(
                f"{self.best_metric_name}_best",
                torch.as_tensor(self.best_metric_value, device=self.device),
                prog_bar=False,
                on_epoch=True,
            )

    def on_train_epoch_end(self) -> None:
        if not self.train_epoch_eval:
            return
        if self.trainer is None:
            return
        train_loader = self._resolve_train_eval_loader()
        if train_loader is None:
            return

        was_training = self.model.training
        self.model.eval()

        loss_sums: Dict[str, torch.Tensor] = {}
        metric_sums: Dict[str, torch.Tensor] = {}
        metric_counts: Dict[str, torch.Tensor] = {}
        total_samples = 0

        with torch.inference_mode():
            for batch in train_loader:
                images, targets = batch
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = self(images)
                loss_dict, _ = self._compute_loss(outputs, targets)

                batch_size = images.size(0)
                total_samples += batch_size
                for name, value in loss_dict.items():
                    loss_sums[name] = loss_sums.get(name, 0) + value.detach() * batch_size

                preds = self._extract_pred_masks(outputs)
                metrics = self._compute_segmentation_metrics(preds, targets)
                for name, value in metrics.items():
                    value_detached = value.detach()
                    metric_sums[name] = metric_sums.get(name, 0) + value_detached.sum()
                    metric_counts[name] = metric_counts.get(name, 0) + value_detached.numel()

        log_payload: Dict[str, torch.Tensor] = {}
        for name, total in loss_sums.items():
            avg = total / max(total_samples, 1)
            log_payload[f"train_epoch/{name}"] = avg
        for name, total in metric_sums.items():
            count = metric_counts[name]
            count = count if count > 1 else 1
            avg = total / count
            log_payload[f"train_epoch/{name}"] = avg
        if log_payload:
            self.log_dict(
                log_payload,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if self.trainer is not None and self.trainer.logger is not None:
                self.trainer.logger.log_metrics(
                    {k: float(v.detach().cpu()) for k, v in log_payload.items()},
                    step=self.global_step,
                )

        if was_training:
            self.model.train()

    def _resolve_train_eval_loader(self) -> Optional[DataLoader]:
        trainer = self.trainer
        if trainer is None:
            return None
        train_loader = getattr(trainer, "train_dataloader", None)
        if callable(train_loader):
            train_loader = train_loader()
        if train_loader is None:
            fit_loop = getattr(trainer, "fit_loop", None)
            if fit_loop is not None:
                train_loader = getattr(fit_loop, "dataloader", None)
                if train_loader is None:
                    train_loader = getattr(fit_loop, "_combined_loader", None)
        if train_loader is None:
            datamodule = trainer.datamodule
            if datamodule is None:
                return None
            train_loader = datamodule.train_dataloader()
        if train_loader is None:
            return None
        if not isinstance(train_loader, DataLoader):
            return train_loader
        if train_loader.num_workers == 0:
            return train_loader
        self.log(
            "train_epoch/loader_warning",
            torch.as_tensor(1.0, device=self.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=None,
        )
        return DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last,
            collate_fn=train_loader.collate_fn,
        )

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_dicts,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.adamw_betas,
                eps=self.adamw_eps,
            )
        else:
            optimizer = SGD(
                param_dicts,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        max_epochs = max(self.max_epochs, 1)
        if self.scheduler_name == "none":
            return {"optimizer": optimizer}
        if self.scheduler_name == "cosine":
            t_max = self.cosine_t_max or max_epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, t_max)
            )
        elif self.scheduler_name == "cosine_restart":
            t_0 = self.cosine_restart_t_0 or max_epochs
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, t_0),
                T_mult=max(1, self.cosine_restart_t_mult),
                eta_min=self.cosine_restart_eta_min,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** self.poly_power,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _compute_loss(
        self, outputs: Any, targets: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if (
            isinstance(outputs, (list, tuple))
            and len(outputs) == 2
            and torch.is_tensor(outputs[0])
        ):
            logits, guide_mask = outputs
            try:
                raw_loss = self.criterion(logits, targets, guide_mask)
            except TypeError:
                raw_loss = self.criterion(logits, targets)
        else:
            raw_loss = self.criterion(outputs, targets)
        if isinstance(raw_loss, dict):
            tensor_losses = {k: self._to_tensor(v) for k, v in raw_loss.items()}
            total_loss = torch.stack(list(tensor_losses.values())).sum()
            return tensor_losses, total_loss
        tensor_loss = self._to_tensor(raw_loss)
        return {"loss": tensor_loss}, tensor_loss

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _extract_pred_masks(self, outputs: Any) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, dict):
            for key in ("pred_masks", "masks", "logits", "out"):
                tensor = outputs.get(key)
                if torch.is_tensor(tensor):
                    return tensor
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                try:
                    return self._extract_pred_masks(item)
                except ValueError:
                    continue
        raise ValueError("Unable to extract prediction tensor from model outputs.")

    def _compute_segmentation_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-6
        preds = preds.detach()
        targets = targets.detach().float()
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        preds_bin = self._binarize_predictions(preds)
        targets_bin = (targets > 0.5).float()

        intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
        pred_area = preds_bin.sum(dim=(1, 2, 3))
        target_area = targets_bin.sum(dim=(1, 2, 3))
        union = pred_area + target_area - intersection

        dice = (2 * intersection + eps) / (pred_area + target_area + eps)
        jaccard = (intersection + eps) / (union + eps)

        hd95_list: List[float] = []
        asd_list: List[float] = []
        preds_np = preds_bin.cpu().numpy()
        targets_np = targets_bin.cpu().numpy()
        for pred_mask, target_mask in zip(preds_np, targets_np):
            hd95, asd = self._surface_metrics(pred_mask[0], target_mask[0])
            hd95_list.append(hd95)
            asd_list.append(asd)

        device = preds.device
        metrics = {
            "dice": dice.mean(),
            "jaccard": jaccard.mean(),
            "hd95": torch.as_tensor(np.mean(hd95_list), device=device, dtype=torch.float32),
            "asd": torch.as_tensor(np.mean(asd_list), device=device, dtype=torch.float32),
        }
        return metrics

    def _binarize_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 4 and preds.size(1) > 1:
            probs = torch.softmax(preds, dim=1)
            binary = torch.argmax(probs, dim=1, keepdim=True).float()
            return binary
        probs = torch.sigmoid(preds)
        return (probs > self.threshold).float()

    def _surface_metrics(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        if pred_bool.sum() == 0 and target_bool.sum() == 0:
            return 0.0, 0.0
        if pred_bool.sum() == 0 or target_bool.sum() == 0:
            diag = float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))
            return diag, diag

        pred_border = pred_bool ^ binary_erosion(pred_bool)
        target_border = target_bool ^ binary_erosion(target_bool)

        dt_pred = distance_transform_edt(~pred_bool)
        dt_target = distance_transform_edt(~target_bool)

        surface_distances: List[np.ndarray] = []
        dist_target_to_pred = dt_pred[target_border]
        if dist_target_to_pred.size:
            surface_distances.append(dist_target_to_pred)
        dist_pred_to_target = dt_target[pred_border]
        if dist_pred_to_target.size:
            surface_distances.append(dist_pred_to_target)

        if not surface_distances:
            return 0.0, 0.0

        all_distances = np.concatenate(surface_distances)
        hd95 = float(np.percentile(all_distances, 95))
        asd = float(np.mean(all_distances))
        return hd95, asd

    def _should_log_images(self, batch_idx: int) -> bool:
        if self.log_image_samples <= 0:
            return False
        if batch_idx != 0:
            return False
        if (self.current_epoch % self.log_image_every_n_epochs) != 0:
            return False
        if getattr(self, "global_rank", 0) != 0:
            return False
        if self.trainer is None or self.trainer.logger is None:
            return False
        return True

    def _log_images(self, images: torch.Tensor, targets: torch.Tensor, outputs: Any) -> None:
        try:
            import wandb
        except Exception:
            return

        logger = self.trainer.logger
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            return

        preds = self._extract_pred_masks(outputs)
        preds_bin = self._binarize_predictions(preds)
        guide_mask = None
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2 and torch.is_tensor(outputs[1]):
            guide_mask = outputs[1]

        batch_size = images.size(0)
        num_samples = min(self.log_image_samples, batch_size)
        indices = torch.randperm(batch_size, device=images.device)[:num_samples]

        logged = []
        for i in indices.tolist():
            img = images[i].detach().float().cpu()
            if img.dim() == 3 and img.size(0) in (1, 3):
                img = img.permute(1, 2, 0)  # HWC
            img_min = float(img.min())
            img_max = float(img.max())
            img = (img - img_min) / (img_max - img_min + 1e-6)
            if img.dim() == 2:
                img = img.unsqueeze(-1)
            if img.size(-1) == 1:
                img = img.repeat(1, 1, 3)

            pred_mask = preds_bin[i].detach().cpu()
            if pred_mask.dim() == 3:
                pred_mask = pred_mask[0]
            gt_mask = targets[i].detach().float().cpu()
            if gt_mask.dim() == 3:
                gt_mask = gt_mask[0]

            masks = {
                "prediction": {"mask_data": pred_mask.numpy()},
                "ground_truth": {"mask_data": gt_mask.numpy()},
            }

            caption = f"epoch {self.current_epoch} sample {i}"
            image = wandb.Image(img.numpy(), masks=masks, caption=caption)

            if guide_mask is not None:
                guide = guide_mask[i].detach().float().cpu()
                if guide.dim() == 3:
                    guide = guide[0]
                guide_img = wandb.Image(
                    guide.numpy(),
                    caption=f"guide {caption}",
                )
                logged.append({"image": image, "guide": guide_img})
            else:
                logged.append({"image": image})

        if logged:
            experiment.log(
                {"val/visuals": logged},
                step=self.global_step,
            )
