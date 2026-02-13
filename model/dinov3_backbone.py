from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel


DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query",
    "key",
    "value",
    "dense",
    "fc1",
    "fc2",
)


class DINOv3BackboneWrapper(nn.Module):
    """
    Wrap a DINOv3 ViT-like model to output a 2D feature map (B, C, H, W).
    Expects a forward that returns token embeddings with an optional CLS token.
    """

    def __init__(
        self,
        backbone: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        train_backbone: bool = False,
        lora_enable: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        lora_bias: str = "none",
        lora_task_type: str = "FEATURE_EXTRACTION",
    ):
        super().__init__()
        self.train_backbone = bool(train_backbone)
        self.lora_enable = bool(lora_enable)
        self.lora_target_modules_requested = list(
            lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
        )
        if self.lora_enable and not self.lora_target_modules_requested:
            raise ValueError("LoRA is enabled but lora_target_modules is empty.")
        self.lora_target_modules_effective: list[str] = []
        self.lora_task_type = str(lora_task_type)
        self._lora_train_heads = True

        self.backbone = AutoModel.from_pretrained(
            backbone,
            device_map="auto",
        )
        self.embed_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = getattr(self.backbone.config, "num_register_tokens", 0)

        if self.lora_enable:
            self._apply_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=self.lora_target_modules_requested,
                bias=lora_bias,
                task_type=self.lora_task_type,
            )
            self.enable_lora_training(train_base_backbone=self.train_backbone, train_heads=True)
        elif not self.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 1x1 conv to map backbone dim -> transformer hidden dim will be added in the head
        # (kept out here to keep this wrapper generic)

    def _resolve_lora_target_modules(self, requested: Sequence[str]) -> list[str]:
        linear_module_names = [
            name for name, module in self.backbone.named_modules() if isinstance(module, nn.Linear)
        ]
        matched: list[str] = []
        for target in requested:
            target = str(target).strip()
            if not target:
                continue
            has_match = any(name == target or name.endswith(f".{target}") for name in linear_module_names)
            if has_match:
                matched.append(target)
        deduped = sorted(set(matched))
        if deduped:
            return deduped

        candidates = sorted({name.split(".")[-1] for name in linear_module_names})
        candidate_preview = ", ".join(candidates[:20]) if candidates else "(no linear modules found)"
        raise ValueError(
            "None of the requested LoRA target modules matched this backbone. "
            f"Requested={list(requested)}. Linear module suffix candidates: {candidate_preview}"
        )

    def _apply_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: Sequence[str],
        bias: str,
        task_type: str,
    ) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA was requested but PEFT is not installed. Install with `pip install peft`."
            ) from exc
        resolved_task_type = getattr(TaskType, task_type.upper(), None)
        if resolved_task_type is None:
            valid = ", ".join(sorted(TaskType.__members__.keys()))
            raise ValueError(f"Unknown LoRA task type '{task_type}'. Valid task types: {valid}")
        resolved_targets = self._resolve_lora_target_modules(target_modules)
        lora_cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=resolved_targets,
            bias=str(bias),
            task_type=resolved_task_type,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        self.lora_target_modules_effective = resolved_targets

    def enable_lora_training(self, train_base_backbone: bool, train_heads: bool) -> None:
        self.train_backbone = bool(train_base_backbone)
        self._lora_train_heads = bool(train_heads)
        if not self.lora_enable:
            for param in self.backbone.parameters():
                param.requires_grad = self.train_backbone
            return
        for param in self.backbone.parameters():
            param.requires_grad = self.train_backbone
        if not self.train_backbone:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def load_lora_adapter(self, adapter_path: str | Path) -> None:
        adapter_path = Path(adapter_path).expanduser()
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Cannot load LoRA adapter because PEFT is not installed. Install with `pip install peft`."
            ) from exc

        if self.lora_enable and hasattr(self.backbone, "load_adapter"):
            adapter_name = "external"
            self.backbone.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
            if hasattr(self.backbone, "set_adapter"):
                self.backbone.set_adapter(adapter_name)
        else:
            self.backbone = PeftModel.from_pretrained(self.backbone, str(adapter_path), is_trainable=False)
            self.lora_enable = True
        self.enable_lora_training(train_base_backbone=False, train_heads=True)

    def print_trainable_summary(self, prefix: str = "DINOv3Backbone") -> dict[str, Any]:
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        summary = {
            "lora_enable": self.lora_enable,
            "lora_target_modules_effective": list(self.lora_target_modules_effective),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "train_base_backbone": self.train_backbone,
        }
        effective = ", ".join(summary["lora_target_modules_effective"]) or "(none)"
        print(
            f"{prefix}: lora_enable={summary['lora_enable']} "
            f"trainable_params={summary['trainable_params']}/{summary['total_params']} "
            f"base_backbone_trainable={summary['train_base_backbone']} "
            f"lora_targets={effective}"
        )
        if self.lora_enable and hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()
        return summary

    def _strip_special_tokens(self, x, img_h, img_w):
        tokens = (img_h // self.patch_size) * (img_w // self.patch_size)
        if x.shape[1] > tokens:
            if self.num_register_tokens > 0:
                return x[:, 1:-self.num_register_tokens, :]
            return x[:, 1:, :]
        return x

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
        backbone_out = self._forward_backbone(images, output_hidden_states=False)
        tokens = backbone_out.last_hidden_state
        feat = self._tokens_to_map(tokens, H, W)  # (B, C, H/ps, W/ps)
        mask = torch.zeros(
            (B, feat.shape[-2], feat.shape[-1]), dtype=torch.bool, device=feat.device)
        return feat, mask

    def get_intermediate_layers(self, images: torch.Tensor, n):
        """
        Return intermediate token features at the specified block indices.
        """
        B, _, H, W = images.shape
        outputs = self._forward_backbone(images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        features = []
        for idx in n:
            state = hidden_states[idx + 1]
            state = self._strip_special_tokens(state, H, W)
            features.append(state)
        return features

    def _forward_backbone(self, images: torch.Tensor, output_hidden_states: bool):
        kwargs = {"output_hidden_states": output_hidden_states}
        try:
            return self.backbone(pixel_values=images, **kwargs)
        except TypeError:
            return self.backbone(images, **kwargs)
