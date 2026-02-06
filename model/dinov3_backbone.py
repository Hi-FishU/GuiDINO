import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Literal


class DINOv3BackboneWrapper(nn.Module):
    """
    Wrap a DINOv3 ViT-like model to output a 2D feature map (B, C, H, W).
    Expects a forward that returns token embeddings with an optional CLS token.
    """

    def __init__(self, backbone: str = 'facebook/dinov3-vit7b16-pretrain-lvd1689m', train_backbone: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone,
            device_map="auto",
        )
        self.embed_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 1x1 conv to map backbone dim -> transformer hidden dim will be added in the head
        # (kept out here to keep this wrapper generic)

    def _strip_special_tokens(self, x, img_h, img_w):
        tokens = (img_h // self.patch_size) * (img_w // self.patch_size)
        if x.shape[1] > tokens:
            if self.num_register_tokens > 0:
                return x[:, 1:-self.num_register_tokens, :]
            return x[:, 1:, :]
        return x

    @torch.no_grad()
    def _tokens_to_map(self, x, img_h, img_w):
        """
        x: (B, HW or 1+HW, C). If CLS present, it's first token.
        Returns feature map (B, C, H, W).
        """
        B, _, C = x.shape
        x = self._strip_special_tokens(x, img_h, img_w)
        H = img_h // self.patch_size
        W = img_w // self.patch_size
        x = x.transpose(1, 2).contiguous()  # (B, C, HW)
        x = x.view(B, C, H, W)              # (B, C, H, W)
        return x

    def forward(self, images: torch.Tensor, sizes=None):
        """
        images: (B, 3, H, W), pixel space already normalized per DINOv3 preproc.
        sizes: optional list of (orig_h, orig_w) for masks; if not provided, uses images.shape.

        Returns:
          feat: (B, C, H', W') feature map
          mask: (B, H', W') padding mask (False where valid). For fully dense images, mask=False.
        """
        B, _, H, W = images.shape
        # Backbone forward – adapt this depending on your DINOv3 API.
        # Common patterns: model.forward_features(images) or model(images, return_all_tokens=True)
        # <- ensure this returns patch tokens (B, 1+HW, C) or (B, HW, C)
        tokens = self.backbone(images).last_hidden_state
        feat = self._tokens_to_map(tokens, H, W)  # (B, C, H/ps, W/ps)
        mask = torch.zeros(
            (B, feat.shape[-2], feat.shape[-1]), dtype=torch.bool, device=feat.device)
        return feat, mask

    def get_intermediate_layers(self, images: torch.Tensor, n):
        """
        Return intermediate token features at the specified block indices.
        """
        B, _, H, W = images.shape
        outputs = self.backbone(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        features = []
        for idx in n:
            state = hidden_states[idx + 1]
            state = self._strip_special_tokens(state, H, W)
            features.append(state)
        return features
