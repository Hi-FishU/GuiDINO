
import pytorch_lightning as pl
from torch.optim import AdamW, SGD
import torch

from model.detr import DETRHead
from utils.criterion import SetCriterion, SingleObjectAccuracy
from utils.post_processor import DETRPostProcessor


class DETRLightningModule(pl.LightningModule):
    def __init__(self, detr_head: DETRHead, criterion: SetCriterion, postproc: DETRPostProcessor, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['detr_head', 'criterion', 'postproc'])
        self.model = detr_head
        self.criterion = criterion
        self.postproc = postproc
        self.lr = lr
        self.weight_decay = weight_decay
        self.obj_acc = SingleObjectAccuracy()


    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        bs = len(batch)
        images, targets = batch
        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True,
                     on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        bs = len(batch)
        images, targets = batch
        outputs = self(images)
        loss_dict = self.criterion(outputs, targets)
        loss = sum(loss_dict.values())
        pred = self.postproc(outputs, [t["orig_size"] for t in targets])
        self.obj_acc.update(pred, targets)
        self.log("val/obj_acc", self.obj_acc.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False,
                     on_epoch=True, prog_bar=False, batch_size=bs)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss


    def configure_optimizers(self):
        # param_dicts = [
        #     {"params": [p for n, p in self.model.named_parameters(
        #     ) if p.requires_grad and "backbone" not in n], "lr": self.lr},
        #     {"params": [p for n, p in self.model.named_parameters(
        #     ) if p.requires_grad and "backbone" in n], "lr": self.lr * 0.1},
        # ]
        param_dicts = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]
        optimizer = SGD(param_dicts, lr=self.lr,
                          weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
