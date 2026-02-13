
from __future__ import annotations

import itertools
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
        lr_lora: Optional[float] = None,
        lr_head: Optional[float] = None,
        lr_backbone: Optional[float] = None,
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
        val_use_sliding_window: bool = False,
        val_sw_patch_size: Tuple[int, int] = (512, 512),
        val_sw_overlap: float = 0.5,
        val_sw_mirror: bool = True,
        val_sw_mirror_axes: Tuple[int, ...] = (0, 1),
        val_sw_gaussian_sigma_scale: float = 1.0 / 8.0,
        val_sw_gaussian_value_scaling: float = 10.0,
        eval_keep_largest_component: bool = False,
        surface_metric_spacing: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.lr_lora = lr_lora
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
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
        self.val_use_sliding_window = bool(val_use_sliding_window)
        self.val_sw_patch_size = (int(val_sw_patch_size[0]), int(val_sw_patch_size[1]))
        self.val_sw_overlap = float(val_sw_overlap)
        self.val_sw_mirror = bool(val_sw_mirror)
        self.val_sw_mirror_axes = tuple(sorted(set(int(a) for a in val_sw_mirror_axes)))
        self.val_sw_gaussian_sigma_scale = float(val_sw_gaussian_sigma_scale)
        self.val_sw_gaussian_value_scaling = float(val_sw_gaussian_value_scaling)
        self.eval_keep_largest_component = bool(eval_keep_largest_component)
        self.surface_metric_spacing = surface_metric_spacing
        self._val_gaussian_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}

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
            if name == "loss":
                continue
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=images.size(0),
            )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        if not torch.isfinite(loss):
            self.log(
                "train/non_finite_loss",
                torch.as_tensor(1.0, device=self.device),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=images.size(0),
            )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        if self.val_use_sliding_window:
            pred_masks = self._sliding_window_predict_batch(images)
            loss_dict, loss = self._compute_loss(pred_masks, targets)
            outputs: Any = pred_masks
        else:
            outputs = self(images, targets)
            loss_dict, loss = self._compute_loss(outputs, targets)
            pred_masks = self._extract_pred_masks(outputs)
        metrics = self._compute_segmentation_metrics(pred_masks, targets)

        for name, value in loss_dict.items():
            if name == "loss":
                continue
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

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
        if not torch.isfinite(loss):
            self.log(
                "val/non_finite_loss",
                torch.as_tensor(1.0, device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=images.size(0),
            )
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
        named_trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if not named_trainable:
            raise RuntimeError("No trainable parameters found. Check model freezing or LoRA settings.")

        use_split_lrs = any(v is not None for v in (self.lr_lora, self.lr_head, self.lr_backbone))
        if not use_split_lrs:
            param_dicts = [{"params": [p for _, p in named_trainable], "lr": self.lr}]
        else:
            lora_params: list[torch.nn.Parameter] = []
            backbone_params: list[torch.nn.Parameter] = []
            head_params: list[torch.nn.Parameter] = []

            for name, param in named_trainable:
                if "lora_" in name:
                    lora_params.append(param)
                    continue
                is_backbone = (
                    "guide_backbone" in name
                    or name.startswith("backbone.")
                    or ".backbone." in f".{name}."
                )
                if is_backbone:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            param_dicts = []
            if lora_params:
                param_dicts.append(
                    {"params": lora_params, "lr": self.lr_lora if self.lr_lora is not None else self.lr}
                )
            if backbone_params:
                param_dicts.append(
                    {"params": backbone_params, "lr": self.lr_backbone if self.lr_backbone is not None else self.lr}
                )
            if head_params:
                param_dicts.append(
                    {"params": head_params, "lr": self.lr_head if self.lr_head is not None else self.lr}
                )

            print(
                "Optimizer LR groups: "
                f"lora={len(lora_params)} params @ {self.lr_lora if self.lr_lora is not None else self.lr}, "
                f"backbone={len(backbone_params)} params @ {self.lr_backbone if self.lr_backbone is not None else self.lr}, "
                f"head={len(head_params)} params @ {self.lr_head if self.lr_head is not None else self.lr}"
            )

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_dicts,
                weight_decay=self.weight_decay,
                betas=self.adamw_betas,
                eps=self.adamw_eps,
            )
        else:
            optimizer = SGD(
                param_dicts,
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
            and (
                torch.is_tensor(outputs[0])
                or (
                    isinstance(outputs[0], (list, tuple))
                    and len(outputs[0]) > 0
                    and all(torch.is_tensor(o) for o in outputs[0])
                )
            )
        ):
            logits, guide_mask = outputs
            if isinstance(logits, (list, tuple)):
                ds_losses = self._compute_deep_supervision_loss(logits, targets, guide_mask=None)
                main_logits = logits[0]
                try:
                    guide_total = self._to_tensor(self.criterion(main_logits, targets, guide_mask))
                except TypeError:
                    guide_total = self._to_tensor(self.criterion(main_logits, targets))
                # Keep DS optimization while adding only the guide-specific part from the main head.
                ds_main = ds_losses.get("ds_loss_0")
                if ds_main is not None:
                    ds_main = self._to_tensor(ds_main)
                    guide_only = guide_total - ds_main
                    ds_losses["guide_loss"] = guide_only
                    total_ds_loss = torch.stack(list(ds_losses.values())).sum()
                    ds_losses["loss"] = total_ds_loss
                    return ds_losses, total_ds_loss
                raw_loss = guide_total
            else:
                try:
                    raw_loss = self.criterion(logits, targets, guide_mask)
                except TypeError:
                    raw_loss = self.criterion(logits, targets)
        elif (
            isinstance(outputs, (list, tuple))
            and len(outputs) > 0
            and all(torch.is_tensor(o) for o in outputs)
        ):
            ds_losses = self._compute_deep_supervision_loss(outputs, targets)
            total_ds_loss = torch.stack(list(ds_losses.values())).sum()
            ds_losses["loss"] = total_ds_loss
            return ds_losses, total_ds_loss
        else:
            try:
                raw_loss = self.criterion(outputs, targets)
            except TypeError:
                raw_loss = self.criterion(outputs, targets, None)
        if isinstance(raw_loss, dict):
            tensor_losses = {k: self._to_tensor(v) for k, v in raw_loss.items()}
            total_loss = torch.stack(list(tensor_losses.values())).sum()
            return tensor_losses, total_loss
        tensor_loss = self._to_tensor(raw_loss)
        return {"loss": tensor_loss}, tensor_loss

    def _compute_deep_supervision_loss(
        self,
        outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        guide_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        num_outputs = len(outputs)
        weights = [1.0 / (2 ** i) for i in range(num_outputs)]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        loss_dict: Dict[str, torch.Tensor] = {}
        for idx, (logits, weight) in enumerate(zip(outputs, weights)):
            target_level = self._resize_target_to_logits(targets, logits)
            try:
                raw_component = self.criterion(logits, target_level, guide_mask)
            except TypeError:
                raw_component = self.criterion(logits, target_level)
            component = self._to_tensor(raw_component) * weight
            loss_dict[f"ds_loss_{idx}"] = component
        return loss_dict

    def _resize_target_to_logits(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if targets.shape[2:] == logits.shape[2:]:
            return targets
        if targets.dtype.is_floating_point:
            resized = torch.nn.functional.interpolate(
                targets.float(),
                size=logits.shape[2:],
                mode="nearest",
            )
            return resized.to(dtype=targets.dtype)
        resized = torch.nn.functional.interpolate(
            targets.float(),
            size=logits.shape[2:],
            mode="nearest",
        )
        return resized.long()

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

    def _sliding_window_predict_batch(self, images: torch.Tensor) -> torch.Tensor:
        preds = [self._sliding_window_predict_single(img.unsqueeze(0)) for img in images]
        return torch.cat(preds, dim=0)

    def _sliding_window_predict_single(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        patch_h, patch_w = self.val_sw_patch_size
        stride_h = max(int(patch_h * (1 - self.val_sw_overlap)), 1)
        stride_w = max(int(patch_w * (1 - self.val_sw_overlap)), 1)

        pad_h = max(patch_h - height, 0)
        pad_w = max(patch_w - width, 0)
        if pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
            height = image.shape[2]
            width = image.shape[3]

        gaussian = self._val_gaussian_map(
            patch_h=patch_h,
            patch_w=patch_w,
            device=image.device,
            dtype=image.dtype,
        )

        output: Optional[torch.Tensor] = None
        weight_sum = torch.zeros((1, 1, height, width), device=image.device, dtype=image.dtype)

        y_positions = list(range(0, height - patch_h + 1, stride_h))
        x_positions = list(range(0, width - patch_w + 1, stride_w))
        if y_positions[-1] != height - patch_h:
            y_positions.append(height - patch_h)
        if x_positions[-1] != width - patch_w:
            x_positions.append(width - patch_w)

        for y in y_positions:
            for x in x_positions:
                patch = image[:, :, y : y + patch_h, x : x + patch_w]
                pred = self._apply_val_mirror(patch)
                if output is None:
                    output = torch.zeros(
                        (1, pred.shape[1], height, width),
                        device=image.device,
                        dtype=pred.dtype,
                    )
                output[:, :, y : y + patch_h, x : x + patch_w] += pred * gaussian
                weight_sum[:, :, y : y + patch_h, x : x + patch_w] += gaussian

        assert output is not None
        output = output / weight_sum.clamp_min(1e-8)
        return output[:, :, : image.shape[2] - pad_h, : image.shape[3] - pad_w]

    def _apply_val_mirror(self, patch: torch.Tensor) -> torch.Tensor:
        if not self.val_sw_mirror:
            pred = self.model(patch)
            return self._extract_pred_masks(pred)

        flips = [()]
        if self.val_sw_mirror_axes:
            for r in range(1, len(self.val_sw_mirror_axes) + 1):
                flips.extend(itertools.combinations(self.val_sw_mirror_axes, r))

        outputs = []
        for axes in flips:
            augmented = patch
            if axes:
                augmented = torch.flip(augmented, dims=[a + 2 for a in axes])
            pred = self.model(augmented)
            pred = self._extract_pred_masks(pred)
            if axes:
                pred = torch.flip(pred, dims=[a + 2 for a in axes])
            outputs.append(pred)
        return torch.stack(outputs, dim=0).mean(dim=0)

    def _val_gaussian_map(
        self,
        patch_h: int,
        patch_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (
            patch_h,
            patch_w,
            str(device),
            str(dtype),
            self.val_sw_gaussian_sigma_scale,
            self.val_sw_gaussian_value_scaling,
        )
        cached = self._val_gaussian_cache.get(key)
        if cached is not None:
            return cached

        yy, xx = torch.meshgrid(
            torch.arange(patch_h, device=device, dtype=dtype),
            torch.arange(patch_w, device=device, dtype=dtype),
            indexing="ij",
        )
        center_y = (patch_h - 1) / 2.0
        center_x = (patch_w - 1) / 2.0
        sigma_y = max(float(patch_h) * self.val_sw_gaussian_sigma_scale, 1e-6)
        sigma_x = max(float(patch_w) * self.val_sw_gaussian_sigma_scale, 1e-6)
        weight = torch.exp(
            -(
                ((yy - center_y) ** 2) / (2.0 * sigma_y ** 2)
                + ((xx - center_x) ** 2) / (2.0 * sigma_x ** 2)
            )
        )
        weight = weight / weight.max().clamp_min(1e-8)
        weight = weight * self.val_sw_gaussian_value_scaling
        min_nonzero = weight[weight > 0].min().clamp_min(1e-8)
        weight = torch.clamp(weight, min=min_nonzero)
        self._val_gaussian_cache[key] = weight
        return weight

    def _compute_segmentation_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        eps = 1e-6
        preds = preds.detach()
        targets = targets.detach()
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if preds.size(1) > 1 or targets.max().item() > 1:
            if preds.size(1) > 1:
                pred_labels = torch.argmax(preds, dim=1)
                num_classes = preds.size(1)
            else:
                pred_labels = (torch.sigmoid(preds) > self.threshold).long().squeeze(1)
                num_classes = int(targets.max().item()) + 1

            if targets.size(1) > 1:
                target_labels = torch.argmax(targets, dim=1)
                num_classes = max(num_classes, targets.size(1))
            else:
                target_labels = targets[:, 0].long()
                num_classes = max(num_classes, int(targets.max().item()) + 1)

            metrics: Dict[str, torch.Tensor] = {}
            dice_values: List[torch.Tensor] = []
            jaccard_values: List[torch.Tensor] = []

            for class_idx in range(num_classes):
                pred_c = pred_labels == class_idx
                target_c = target_labels == class_idx
                intersection = (pred_c & target_c).sum(dim=(1, 2)).float()
                pred_area = pred_c.sum(dim=(1, 2)).float()
                target_area = target_c.sum(dim=(1, 2)).float()
                union = pred_area + target_area - intersection

                dice = (2 * intersection + eps) / (pred_area + target_area + eps)
                jaccard = (intersection + eps) / (union + eps)
                valid = (pred_area + target_area) > 0
                if valid.any():
                    dice_mean = dice[valid].mean()
                    jaccard_mean = jaccard[valid].mean()
                else:
                    # Class absent in both prediction and target across the batch.
                    # Do not treat this as perfect score.
                    dice_mean = torch.tensor(0.0, device=preds.device)
                    jaccard_mean = torch.tensor(0.0, device=preds.device)

                metrics[f"dice_class_{class_idx}"] = dice_mean
                metrics[f"jaccard_class_{class_idx}"] = jaccard_mean
                if class_idx != 0:
                    if valid.any():
                        dice_values.append(dice_mean)
                        jaccard_values.append(jaccard_mean)

            if dice_values:
                metrics["dice"] = torch.stack(dice_values).mean()
                metrics["jaccard"] = torch.stack(jaccard_values).mean()
            else:
                metrics["dice"] = metrics.get("dice_class_0", torch.tensor(0.0, device=preds.device))
                metrics["jaccard"] = metrics.get("jaccard_class_0", torch.tensor(0.0, device=preds.device))
            return metrics

        preds_bin = self._binarize_predictions(preds)
        targets_bin = (targets > 0.5).float()
        if self.eval_keep_largest_component:
            preds_bin = self._keep_largest_component_batch(preds_bin)

        intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
        pred_area = preds_bin.sum(dim=(1, 2, 3))
        target_area = targets_bin.sum(dim=(1, 2, 3))
        union = pred_area + target_area - intersection

        dice = (2 * intersection + eps) / (pred_area + target_area + eps)
        jaccard = (intersection + eps) / (union + eps)

        hd95_list: List[float] = []
        asd_list: List[float] = []
        hd95_nonempty_list: List[float] = []
        asd_nonempty_list: List[float] = []
        preds_np = preds_bin.cpu().numpy()
        targets_np = targets_bin.cpu().numpy()
        for pred_mask, target_mask in zip(preds_np, targets_np):
            hd95, asd = self._surface_metrics(
                pred_mask[0],
                target_mask[0],
                spacing=self.surface_metric_spacing,
            )
            hd95_list.append(hd95)
            asd_list.append(asd)
            pred_has_fg = bool(pred_mask[0].astype(bool).sum() > 0)
            target_has_fg = bool(target_mask[0].astype(bool).sum() > 0)
            if pred_has_fg and target_has_fg:
                hd95_nonempty_list.append(hd95)
                asd_nonempty_list.append(asd)

        device = preds.device
        hd95_all = float(np.mean(hd95_list)) if hd95_list else 0.0
        asd_all = float(np.mean(asd_list)) if asd_list else 0.0
        if hd95_nonempty_list:
            hd95_nonempty = float(np.mean(hd95_nonempty_list))
            asd_nonempty = float(np.mean(asd_nonempty_list))
        else:
            hd95_nonempty = hd95_all
            asd_nonempty = asd_all
        metrics = {
            "dice": dice.mean(),
            "jaccard": jaccard.mean(),
            "hd95": torch.as_tensor(hd95_nonempty, device=device, dtype=torch.float32),
            "asd": torch.as_tensor(asd_nonempty, device=device, dtype=torch.float32),
            "hd95_all": torch.as_tensor(hd95_all, device=device, dtype=torch.float32),
            "asd_all": torch.as_tensor(asd_all, device=device, dtype=torch.float32),
            "hd95_nonempty": torch.as_tensor(hd95_nonempty, device=device, dtype=torch.float32),
            "asd_nonempty": torch.as_tensor(asd_nonempty, device=device, dtype=torch.float32),
            "surface_nonempty_count": torch.as_tensor(float(len(hd95_nonempty_list)), device=device, dtype=torch.float32),
        }
        return metrics

    def _binarize_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 4 and preds.size(1) > 1:
            probs = torch.softmax(preds, dim=1)
            binary = torch.argmax(probs, dim=1, keepdim=True).float()
            return binary
        probs = torch.sigmoid(preds)
        return (probs > self.threshold).float()

    def _keep_largest_component_batch(self, preds_bin: torch.Tensor) -> torch.Tensor:
        if preds_bin.dim() != 4 or preds_bin.size(1) != 1:
            return preds_bin
        preds_np = preds_bin.detach().cpu().numpy().astype(np.uint8)
        processed: List[np.ndarray] = []
        for pred in preds_np:
            processed.append(self._keep_largest_component_2d(pred[0])[None, ...])
        processed_np = np.stack(processed, axis=0).astype(np.float32)
        return torch.from_numpy(processed_np).to(device=preds_bin.device, dtype=preds_bin.dtype)

    def _keep_largest_component_2d(self, mask: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import label
        except Exception:
            return mask.astype(np.uint8)
        labeled, num = label(mask.astype(bool))
        if num <= 1:
            return mask.astype(np.uint8)
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        keep = counts.argmax()
        return (labeled == keep).astype(np.uint8)

    def _surface_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        if pred_bool.sum() == 0 and target_bool.sum() == 0:
            return 0.0, 0.0
        if pred_bool.sum() == 0 or target_bool.sum() == 0:
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
        target_border = target_bool ^ binary_erosion(target_bool)

        dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
        dt_target = distance_transform_edt(~target_bool, sampling=spacing)

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
