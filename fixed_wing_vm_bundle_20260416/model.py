from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalization(
    num_channels: int,
    norm_type: str,
    group_norm_groups: int,
) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "groupnorm":
        groups = max(1, min(group_norm_groups, num_channels))
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)

    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_type: str = "batchnorm",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_normalization(out_channels, norm_type, group_norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_normalization(out_channels, norm_type, group_norm_groups),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_type: str = "batchnorm",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            dropout=dropout,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "batchnorm",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class LightweightUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
        norm_type: str = "batchnorm",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = ConvBlock(
            in_channels,
            c1,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.down1 = DownBlock(
            c1,
            c2,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.down2 = DownBlock(
            c2,
            c3,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.down3 = DownBlock(
            c3,
            c4,
            dropout=dropout,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )

        self.up2 = UpBlock(
            c4,
            c3,
            c3,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.up1 = UpBlock(
            c3,
            c2,
            c2,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.up0 = UpBlock(
            c2,
            c1,
            c1,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        return self.head(x)
