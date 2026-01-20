
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.optim import SGD


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
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold

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
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.size(0),
        )
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
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]
        optimizer = SGD(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
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
