from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


BNNorm2d = nn.BatchNorm2d
Activation = nn.GELU


class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm2d(out_ch),
            Activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DownConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm2d(out_ch),
            Activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ResBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, groups: int = 1) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = BNNorm2d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn2 = BNNorm2d(planes)
        self.down = None
        if inplanes != planes:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
                BNNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.down is not None:
            identity = self.down(x)
        out = self.bn2(out) + identity
        out = self.act(out)
        return out


class OPE(nn.Module):
    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = BNNorm2d(inplanes)
        self.act = Activation()
        self.down = DownConv(inplanes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.down(out)
        return out


class LocalBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        hidden_planes: int,
        planes: int,
        groups: int = 1,
        down_or_up: str | None = None,
    ) -> None:
        super().__init__()
        if down_or_up is None:
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
            )
        elif down_or_up == "down":
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                DownConv(hidden_planes, planes),
            )
        elif down_or_up == "up":
            self.block = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                UpConv(hidden_planes, planes),
            )
        else:
            raise ValueError(f"Unknown down_or_up option: {down_or_up}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Pooling(nn.Module):
    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(1, num_channels, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = Activation()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        num_heads: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.proj = nn.Conv2d(in_dim, dim, kernel_size=3, padding=1)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = GroupNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WNet2D(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_classes: int,
        deep_supervised: bool = False,
        layer_channel: List[int] | None = None,
        global_dim: List[int] | None = None,
        num_heads: List[int] | None = None,
        sr_ratio: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.deep_supervised = deep_supervised
        layer_channel = layer_channel or [16, 32, 64, 128, 256]
        global_dim = global_dim or [8, 16, 32, 64, 128]
        num_heads = num_heads or [1, 2, 4, 8]
        sr_ratio = sr_ratio or [8, 4, 2, 1]

        self.input_l0 = nn.Sequential(
            nn.Conv2d(in_channel, layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm2d(layer_channel[0]),
            Activation(),
            nn.Conv2d(layer_channel[0], layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm2d(layer_channel[0]),
            Activation(),
        )

        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = GlobalBlock(layer_channel[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = GlobalBlock(layer_channel[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = GlobalBlock(layer_channel[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = GlobalBlock(layer_channel[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder1_l4_local = LocalBlock(layer_channel[4], layer_channel[4], layer_channel[3], down_or_up="up")
        self.decoder1_l4_global = GlobalBlock(layer_channel[4], global_dim[4], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder1_l3_local = LocalBlock(
            layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up="up"
        )
        self.decoder1_l3_global = GlobalBlock(
            layer_channel[3] + global_dim[3], global_dim[3], num_heads=num_heads[2], sr_ratio=sr_ratio[2]
        )

        self.decoder1_l2_local = LocalBlock(
            layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up="up"
        )
        self.decoder1_l2_global = GlobalBlock(
            layer_channel[2] + global_dim[2], global_dim[2], num_heads=num_heads[1], sr_ratio=sr_ratio[1]
        )

        self.decoder1_l1_local = LocalBlock(
            layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up="up"
        )
        self.decoder1_l1_global = GlobalBlock(
            layer_channel[1] + global_dim[1], global_dim[1], num_heads=num_heads[0], sr_ratio=sr_ratio[0]
        )

        self.encoder2_l1_local = LocalBlock(
            layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[1], down_or_up="down"
        )
        self.encoder2_l1_global = GlobalBlock(
            layer_channel[0] + global_dim[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0]
        )

        self.encoder2_l2_local = LocalBlock(
            layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[2], down_or_up="down"
        )
        self.encoder2_l2_global = GlobalBlock(
            layer_channel[1] + global_dim[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1]
        )

        self.encoder2_l3_local = LocalBlock(
            layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[3], down_or_up="down"
        )
        self.encoder2_l3_global = GlobalBlock(
            layer_channel[2] + global_dim[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2]
        )

        self.encoder2_l4_local = LocalBlock(
            layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[4], down_or_up="down"
        )
        self.encoder2_l4_global = GlobalBlock(
            layer_channel[3] + global_dim[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3]
        )

        self.decoder2_l4_local_output = nn.Conv2d(layer_channel[4], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l4_local = LocalBlock(
            layer_channel[4] + global_dim[4], layer_channel[4], layer_channel[3], down_or_up="up"
        )

        self.decoder2_l3_local_output = nn.Conv2d(layer_channel[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l3_local = LocalBlock(
            layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up="up"
        )

        self.decoder2_l2_local_output = nn.Conv2d(layer_channel[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l2_local = LocalBlock(
            layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up="up"
        )

        self.decoder2_l1_local_output = nn.Conv2d(layer_channel[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l1_local = LocalBlock(
            layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up="up"
        )

        self.output_l0 = nn.Sequential(
            LocalBlock(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[0], down_or_up=None),
            nn.Conv2d(layer_channel[0], num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        outputs = []

        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)
        x_d1_l4_global = self.decoder1_l4_global(x_e1_l4_local)

        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        x_d1_l3_global = self.decoder1_l3_global(x_d1_l3)

        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        x_d1_l2_global = self.decoder1_l2_global(x_d1_l2)

        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
        x_d1_l1_global = self.decoder1_l1_global(x_d1_l1)

        x_e2_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        x_e2_l1_local = self.encoder2_l1_local(x_e2_l0)
        x_e2_l0_global = self.encoder2_l1_global(x_e2_l0)

        x_e2_l1 = torch.cat((x_e2_l1_local, x_d1_l1_global), dim=1)
        x_e2_l2_local = self.encoder2_l2_local(x_e2_l1)
        x_e2_l1_global = self.encoder2_l2_global(x_e2_l1)

        x_e2_l2 = torch.cat((x_e2_l2_local, x_d1_l2_global), dim=1)
        x_e2_l3_local = self.encoder2_l3_local(x_e2_l2)
        x_e2_l2_global = self.encoder2_l3_global(x_e2_l2)

        x_e2_l3 = torch.cat((x_e2_l3_local, x_d1_l3_global), dim=1)
        x_e2_l4_local = self.encoder2_l4_local(x_e2_l3)
        x_e2_l3_global = self.encoder2_l4_global(x_e2_l3)

        outputs.append(self.decoder2_l4_local_output(x_e2_l4_local))
        x_e2_l4 = torch.cat((x_e2_l4_local, x_d1_l4_global), dim=1)
        x_d2_l3_local = self.decoder2_l4_local(x_e2_l4)

        outputs.append(self.decoder2_l3_local_output(x_d2_l3_local))
        x_d2_l3 = torch.cat((x_d2_l3_local, x_e2_l3_global), dim=1)
        x_d2_l2_local = self.decoder2_l3_local(x_d2_l3)

        outputs.append(self.decoder2_l2_local_output(x_d2_l2_local))
        x_d2_l2 = torch.cat((x_d2_l2_local, x_e2_l2_global), dim=1)
        x_d2_l1_local = self.decoder2_l2_local(x_d2_l2)

        outputs.append(self.decoder2_l1_local_output(x_d2_l1_local))
        x_d2_l1 = torch.cat((x_d2_l1_local, x_e2_l1_global), dim=1)
        x_d2_l0_local = self.decoder2_l1_local(x_d2_l1)

        x_d2_l0 = torch.cat((x_d2_l0_local, x_e2_l0_global), dim=1)
        outputs.append(self.output_l0(x_d2_l0))

        outputs = outputs[::-1]
        if self.deep_supervised:
            return outputs
        return outputs[0]
