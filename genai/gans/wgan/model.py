import torch
import torch.nn as nn

from .utils import ConvBlock, DeconvBlock


class Generator(nn.Module):
    def __init__(self, noise_dim: int, num_features: int, num_blocks: int) -> None:
        super(Generator, self).__init__()
        n = num_blocks

        self.unflatten = nn.Unflatten(1, (noise_dim, 1, 1))
        self.initial = DeconvBlock(
            noise_dim, num_features * (2**n), kernel_size=4, stride=1, padding=0
        )

        self.blocks = nn.Sequential(
            *[
                DeconvBlock(
                    num_features * (2 ** (i + 1)),
                    num_features * (2**i),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for i in range(n - 1, 0, -1)
            ]
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(num_features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        channeled_ip = self.unflatten(input)
        output = self.initial(channeled_ip)
        output = self.blocks(output)
        output = self.final(output)

        return output


class Critic(nn.Module):
    def __init__(self, num_features: int, num_blocks: int) -> None:
        super(Critic, self).__init__()
        n = num_blocks

        self.initial = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.blocks = nn.Sequential(
            *[
                ConvBlock(
                    num_features * (2**i),
                    num_features * (2 ** (i + 1)),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
                for i in range(n - 1)
            ]
        )

        self.final = nn.Conv2d(
            num_features * (2 ** (n - 1)), 1, kernel_size=4, stride=2, padding=0
        )
        self.flatten = nn.Flatten()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.initial(input)
        output = self.blocks(output)
        output = self.final(output)

        return self.flatten(output)
