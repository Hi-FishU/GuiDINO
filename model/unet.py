from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_backbone import DINOv3BackboneWrapper
from .tokenbook import TokenBook

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.last_encoded_shape: Optional[Tuple[int, int, int]] = None

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        self.last_encoded_shape = (x5.size(1), x5.size(2), x5.size(3))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_last_encoded_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.last_encoded_shape


class GuideUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
        guide_backbone_name: str | None = None,
        guide_backbone_train: bool = False,
        tokenbook_tokens: int | None = None,
        tokenbook_image_size: int | None = None,
        tokenbook_dropout: float = 0.0,
        tokenbook_sample_rate: float = 1.0,
        tokenbook_ema_decay: float | None = None,
        tokenbook_use_ema: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.last_encoded_shape: Optional[Tuple[int, int, int]] = None
        self.tokenbook_sample_rate = float(tokenbook_sample_rate)
        self.guide_backbone_train = guide_backbone_train

        self.guide_backbone = None
        if guide_backbone_name is not None:
            self.guide_backbone = DINOv3BackboneWrapper(
                backbone=guide_backbone_name,
                train_backbone=guide_backbone_train,
            )

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        bottom_channels = base_channels * 16 // factor
        guide_embed_dim = (
            self.guide_backbone.embed_dim
            if self.guide_backbone is not None
            else bottom_channels
        )
        if tokenbook_tokens is None:
            if tokenbook_image_size is None:
                raise ValueError(
                    "tokenbook_tokens or tokenbook_image_size must be provided."
                )
            if self.guide_backbone is not None:
                patch = self.guide_backbone.patch_size
                if tokenbook_image_size % patch != 0:
                    raise ValueError(
                        f"tokenbook_image_size {tokenbook_image_size} must be divisible by patch_size {patch}."
                    )
                grid = tokenbook_image_size // patch
            else:
                downsample = 2 ** 4
                if tokenbook_image_size % downsample != 0:
                    raise ValueError(
                        f"tokenbook_image_size {tokenbook_image_size} must be divisible by {downsample}."
                    )
                grid = tokenbook_image_size // downsample
            tokenbook_tokens = grid * grid
        self.tokenbook = TokenBook(
            n_tokens=tokenbook_tokens,
            embed_dim=guide_embed_dim,
            dropout=tokenbook_dropout,
            ema_decay=tokenbook_ema_decay,
            use_ema=tokenbook_use_ema,
        )

        self.up1 = Up(base_channels * 16, base_channels *
                      8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, num_classes)

    def forward(
        self, x: torch.Tensor, guide_feat: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if guide_feat is None and self.guide_backbone is not None:
            if self.guide_backbone_train:
                guide_feat, _ = self.guide_backbone(x)
            else:
                with torch.no_grad():
                    guide_feat, _ = self.guide_backbone(x)
        if guide_feat is None:
            guide_feat = x5

        patch_h, patch_w = guide_feat.shape[-2:]
        tokens = guide_feat.flatten(2).transpose(1, 2)
        token_mask = None
        if self.tokenbook_sample_rate < 1.0:
            B, L, _ = tokens.shape
            token_mask = (
                torch.rand(B, L, device=tokens.device) < self.tokenbook_sample_rate
            )
            if token_mask.sum(dim=1).min().item() == 0:
                rand_idx = torch.randint(0, L, (B,), device=tokens.device)
                token_mask[torch.arange(B, device=tokens.device), rand_idx] = True
        guide = self.tokenbook(
            tokens, height=patch_h, width=patch_w, token_mask=token_mask
        )
        guide_for_unet = guide
        if guide_for_unet.shape[-2:] != x5.shape[-2:]:
            guide_for_unet = F.interpolate(
                guide_for_unet,
                size=x5.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x5 = x5 * guide_for_unet

        self.last_encoded_shape = (x5.size(1), x5.size(2), x5.size(3))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, guide

    def get_last_encoded_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.last_encoded_shape
