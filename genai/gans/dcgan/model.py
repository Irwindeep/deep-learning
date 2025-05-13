import torch
import torch.nn as nn

from .utils import ConvBlock, UpsampleBlock


def initialize_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_blocks: int,
    ) -> None:
        super(Discriminator, self).__init__()

        self.down_conv = ConvBlock(
            3,
            num_features,
            kernel_size=4,
            stride=2,
            padding=1,
            act="leaky_relu",
        )
        self.disc = nn.Sequential(
            *[
                ConvBlock(
                    num_features * (2**i),
                    num_features * (2 ** (i + 1)),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    act="leaky_relu",
                )
                for i in range(num_blocks)
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(num_features * (2 ** (num_blocks)), 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.Flatten(),
        )
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv_feats = self.down_conv(input)
        disc_out = self.disc(conv_feats)
        output = self.final(disc_out)

        return output


class Generator(nn.Module):
    def __init__(self, noise_dim: int, num_features: int, num_blocks: int) -> None:
        super(Generator, self).__init__()

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(noise_dim, 1, 1))
        self.initial = UpsampleBlock(
            noise_dim,
            num_features * (2 ** (num_blocks)),
            kernel_size=3,
            stride=2,
        )
        self.conv_net = nn.Sequential(
            *[
                UpsampleBlock(
                    num_features * (2 ** (i + 1)),
                    num_features * (2**i),
                    kernel_size=3,
                    stride=2,
                )
                for i in range(num_blocks - 1, -1, -1)
            ]
        )
        self.final = nn.Sequential(
            UpsampleBlock(num_features, 3, kernel_size=3, stride=2, only_conv=True),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feats = self.initial(self.unflatten(input))
        conv_out = self.conv_net(feats)
        output = self.final(conv_out)

        return output
