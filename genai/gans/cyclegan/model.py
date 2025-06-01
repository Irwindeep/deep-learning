from typing import List
import torch
import torch.nn as nn
from .utils import ConvBlock, UpsampleBlock, ResidualBlock


class Generator(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_blocks: int,
    ) -> None:
        super(Generator, self).__init__()

        self.initial = ConvBlock(3, num_features, kernel_size=7, padding=3)
        self.down_proj = nn.Sequential(
            ConvBlock(num_features, num_features * 2, 4, stride=2, padding=1),
            ConvBlock(num_features * 2, num_features * 4, 4, stride=2, padding=1),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=num_features * 4) for _ in range(num_blocks)]
        )

        self.up_proj = nn.Sequential(
            UpsampleBlock(num_features * 4, num_features * 2, 4, stride=2, padding=1),
            UpsampleBlock(num_features * 2, num_features, 4, stride=2, padding=1),
        )
        self.final = nn.Sequential(
            nn.Conv2d(
                num_features, 3, kernel_size=7, padding=3, padding_mode="reflect"
            ),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial(input)
        output = self.down_proj(output)
        output = self.residual_blocks(output)
        output = self.up_proj(output)
        output = self.final(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, channels: List[int]) -> None:
        super(Discriminator, self).__init__()

        in_channs, out_channs = channels[:-1], channels[1:]
        self.model = nn.Sequential(
            *[
                ConvBlock(
                    in_channels=in_chann,
                    out_channels=out_chann,
                    kernel_size=4,
                    stride=2 if out_chann != out_channs[-1] else 1,
                    padding=1,
                    act="leaky_relu",
                )
                for in_chann, out_chann in zip(in_channs, out_channs)
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(
                out_channs[-1],
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.model(input)
        output = self.final(output)

        return output
