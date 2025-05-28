from typing import Optional

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
    ) -> None:
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding else "same",
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super(DeconvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)


def gradient_penalty(
    critic: nn.Module, X_r: torch.Tensor, X_g: torch.Tensor, device: str
) -> torch.Tensor:
    N, C, H, W = X_r.size()

    alpha = torch.rand((N, 1, 1, 1)).repeat(1, C, H, W).to(device)
    X_i = alpha * X_r + (1 - alpha) * X_g

    critic_i = critic(X_i)
    gradient = torch.autograd.grad(
        outputs=critic_i,
        inputs=X_i,
        grad_outputs=torch.ones_like(critic_i),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = torch.flatten(gradient, start_dim=1)
    grad_norm = gradient.norm(2, dim=1)

    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty
