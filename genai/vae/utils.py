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

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding else "same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: Optional[int] = None,
        mode: str = "nearest",
        only_conv: bool = False,
    ) -> None:
        super(UpsampleBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=stride, mode=mode)
        if not only_conv:
            self.conv = ConvBlock(
                in_channels, out_channels, kernel_size, padding=padding
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding if padding else "same",
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        upsampled_ip = self.upsample(input)
        output = self.conv(upsampled_ip)

        return output


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self)
        self.mse = nn.MSELoss(reduction="sum")

    def forward(
        self,
        X_recon: torch.Tensor,
        X_target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = self.mse(X_recon, X_target)
        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_div_loss
