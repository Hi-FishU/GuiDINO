from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _default_features_per_stage(n_stages: int, base_features: int, max_features: int) -> list[int]:
    return [min(base_features * (2 ** stage), max_features) for stage in range(n_stages)]


def _default_blocks_per_stage(n_stages: int) -> list[int]:
    base = [1, 3, 4, 6, 6, 6, 6]
    if n_stages <= len(base):
        return base[:n_stages]
    return base + [base[-1]] * (n_stages - len(base))


class OfficialNNUNet2D(nn.Module):
    """
    nnUNet-v2-style 2D network wrapper built from official nnUNet utilities.

    This class uses `nnunetv2.utilities.get_network_from_plans.get_network_from_plans`
    so architecture construction follows the official code path.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        deep_supervision: bool = True,
        n_stages: int = 6,
        features_per_stage: Sequence[int] | None = None,
        base_features: int = 32,
        max_features: int = 512,
        architecture_class_name: str = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    ) -> None:
        super().__init__()
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 for nnUNet-style encoder-decoder.")
        if features_per_stage is not None and len(features_per_stage) != n_stages:
            raise ValueError("features_per_stage length must match n_stages.")

        try:
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        except ImportError as exc:
            raise ImportError(
                "Official nnUNet model requires nnUNet v2. Install with `pip install nnunetv2`."
            ) from exc

        per_stage = (
            list(features_per_stage)
            if features_per_stage is not None
            else _default_features_per_stage(n_stages, base_features, max_features)
        )
        arch_kwargs = {
            "n_stages": n_stages,
            "features_per_stage": per_stage,
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3] for _ in range(n_stages)],
            "strides": [[1, 1]] + [[2, 2] for _ in range(n_stages - 1)],
            "n_blocks_per_stage": _default_blocks_per_stage(n_stages),
            "n_conv_per_stage_decoder": [1 for _ in range(n_stages - 1)],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        }

        self.network = get_network_from_plans(
            arch_class_name=architecture_class_name,
            arch_kwargs=arch_kwargs,
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=in_channels,
            output_channels=num_classes,
            allow_init=True,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)
