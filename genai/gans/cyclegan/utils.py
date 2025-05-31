from typing import Optional
from functools import partial
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        padding_mode: str = "reflect",
        act: Optional[str] = "relu",
    ) -> None:
        super(ConvBlock, self).__init__()

        activation = (
            {
                "relu": partial(nn.ReLU, inplace=True),
                "leaky_relu": partial(nn.LeakyReLU, 0.2, inplace=True),
            }.get(act, partial(nn.ReLU, inplace=True))
            if act
            else None
        )

        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding else "same",
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_channels),
        ]
        if activation:
            modules.append(activation())

        self.conv = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        super(UpsampleBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.upconv(input)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, kernel_size=3, padding=1, act=None),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.block(input)
