from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_backbone import DINOv3BackboneWrapper
from .tokenbook import TokenBook


class SimpleSegmentationDecoder(nn.Module):
    """
    Lightweight decoder for DINOv3 feature maps.
    Produces per-pixel logits at the input image resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        num_groups = 32 if hidden_dim % 32 == 0 else 1
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(
        self, feat: torch.Tensor, output_size: Tuple[int, int]
    ) -> torch.Tensor:
        x = self.proj(feat)
        x = self.block(x)
        x = self.classifier(x)
        if output_size is not None:
            x = F.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return x


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    return scratch


class SegDINOHead(nn.Module):
    """
    SegDINO-style lightweight MLP decoder head (DPT variant).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        features: int = 128,
        out_channels: List[int] | Tuple[int, int, int, int] = (96, 192, 384, 768),
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )
        self.scratch = _make_scratch(list(out_channels), features, groups=1)
        self.scratch.stem_transpose = None
        self.scratch.output_conv = nn.Conv2d(
            features * 4, num_classes, kernel_size=1, stride=1, padding=0
        )
        self.proj = nn.ConvTranspose2d(
            features, features, kernel_size=4, stride=4, padding=0, bias=False
        )

    def forward(
        self, out_features: List[torch.Tensor], patch_h: int, patch_w: int
    ) -> torch.Tensor:
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        layer_1_rn = self.proj(layer_1_rn)
        target_hw = layer_1_rn.shape[-2:]
        layer_2_up = F.interpolate(
            layer_2_rn, size=target_hw, mode="bilinear", align_corners=True
        )
        layer_3_up = F.interpolate(
            layer_3_rn, size=target_hw, mode="bilinear", align_corners=True
        )
        layer_4_up = F.interpolate(
            layer_4_rn, size=target_hw, mode="bilinear", align_corners=True
        )
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out


class DINOv3SegmentationModel(nn.Module):
    """
    End-to-end DINOv3 segmentation model: backbone + simple decoder.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        train_backbone: bool = False,
        num_classes: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = DINOv3BackboneWrapper(
            backbone=backbone_name,
            train_backbone=train_backbone,
        )
        self.decoder = SimpleSegmentationDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat, _ = self.backbone(images)
        logits = self.decoder(feat, output_size=images.shape[-2:])
        return logits


class SegDINOModel(nn.Module):
    """
    SegDINO-style segmentation model: DINOv3 backbone + DPT head.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        train_backbone: bool = False,
        num_classes: int = 1,
        encoder_size: str = "base",
        features: int = 128,
        out_channels: List[int] | Tuple[int, int, int, int] = (96, 192, 384, 768),
    ):
        super().__init__()
        self.backbone = DINOv3BackboneWrapper(
            backbone=backbone_name,
            train_backbone=train_backbone,
        )
        self.encoder_size = encoder_size
        self.intermediate_layer_idx = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        if encoder_size not in self.intermediate_layer_idx:
            raise ValueError(f"Unknown encoder_size: {encoder_size}")
        self.head = SegDINOHead(
            num_classes=num_classes,
            in_channels=self.backbone.embed_dim,
            features=features,
            out_channels=list(out_channels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_h = images.shape[-2] // self.backbone.patch_size
        patch_w = images.shape[-1] // self.backbone.patch_size
        features = self.backbone.get_intermediate_layers(
            images, n=self.intermediate_layer_idx[self.encoder_size]
        )
        logits = self.head(features, patch_h, patch_w)
        logits = F.interpolate(
            logits,
            size=(patch_h * self.backbone.patch_size, patch_w * self.backbone.patch_size),
            mode="bilinear",
            align_corners=True,
        )
        return logits

class GuideSegmentationDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        tokenbook_tokens: int = 1,
        tokenbook_dropout: float = 0.0,
        tokenbook_sample_rate: float = 1.0,
    ):
        super().__init__()
        num_groups = 32 if hidden_dim % 32 == 0 else 1
        self.tokenbook = TokenBook(
            n_tokens=tokenbook_tokens,
            embed_dim=in_channels,
            dropout=tokenbook_dropout,
        )
        self.tokenbook_sample_rate = float(tokenbook_sample_rate)
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(
        self, feat: torch.Tensor, output_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)
        token_mask = None
        if self.tokenbook_sample_rate < 1.0:
            B, L, _ = tokens.shape
            token_mask = (torch.rand(B, L, device=tokens.device) < self.tokenbook_sample_rate)
            if token_mask.sum(dim=1).min().item() == 0:
                rand_idx = torch.randint(0, L, (B,), device=tokens.device)
                token_mask[torch.arange(B, device=tokens.device), rand_idx] = True
        guide = self.tokenbook(tokens, height=patch_h, width=patch_w, token_mask=token_mask)
        x = self.proj(feat)
        x = x * guide
        x = self.block(x)
        x = self.classifier(x)
        if output_size is not None:
            x = F.interpolate(
                x, size=output_size, mode="bilinear", align_corners=False
            )
        return x, guide


class GuideDINOModel(nn.Module):
    def __init__(self,
        backbone_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        train_backbone: bool = False,
        num_classes: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        tokenbook_tokens: int | None = None,
        tokenbook_image_size: int | None = None,
        tokenbook_dropout: float = 0.0,
        tokenbook_sample_rate: float = 1.0,
    ):
        super().__init__()
        self.backbone = DINOv3BackboneWrapper(
            backbone=backbone_name,
            train_backbone=train_backbone,
        )
        if tokenbook_tokens is None:
            if tokenbook_image_size is None:
                raise ValueError("tokenbook_tokens or tokenbook_image_size must be provided.")
            patch = self.backbone.patch_size
            if tokenbook_image_size % patch != 0:
                raise ValueError(
                    f"tokenbook_image_size {tokenbook_image_size} must be divisible by patch_size {patch}."
                )
            grid = tokenbook_image_size // patch
            tokenbook_tokens = grid * grid
        self.decoder = GuideSegmentationDecoder(
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            tokenbook_tokens=tokenbook_tokens,
            tokenbook_dropout=tokenbook_dropout,
            tokenbook_sample_rate=tokenbook_sample_rate,
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, _ = self.backbone(images)
        logits, guide_mask = self.decoder(feat, output_size=images.shape[-2:])
        return logits, guide_mask
