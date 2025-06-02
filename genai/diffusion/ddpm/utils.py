import torch
import torch.nn as nn
from typing import List, Tuple


class AdaptiveGroupNorm2d(nn.Module):
    def __init__(self, num_groups: int, embed_dim: int) -> None:
        super(AdaptiveGroupNorm2d, self).__init__()

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_groups)
        self.proj = nn.Linear(in_features=embed_dim, out_features=2 * num_groups)

    def forward(self, input: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(self.proj(time_embed), chunks=2, dim=1)
        output = self.norm(input)
        output = output * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = AdaptiveGroupNorm2d(num_groups=out_channels, embed_dim=embed_dim)
        self.activation = nn.SiLU()

    def forward(self, input: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.norm(output, time_embed)
        output = self.activation(output)

        return output


class Encoder(nn.Module):
    def __init__(
        self, channels: List[int], bottleneck_size: int, embed_dim: int
    ) -> None:
        super(Encoder, self).__init__()

        in_channs, out_channs = channels[:-1], channels[1:]
        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_chann, out_channels=out_chann, embed_dim=embed_dim
                )
                for in_chann, out_chann in zip(in_channs, out_channs)
            ]
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Conv2d(
            in_channels=channels[-1],
            out_channels=bottleneck_size,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, input: torch.Tensor, time_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outputs = []
        for conv_layer in self.conv_layers:
            input = conv_layer(input, time_embed)
            outputs.append(input)
            input = self.max_pool(input)

        output = self.bottleneck(input)
        return output, outputs


class Decoder(nn.Module):
    def __init__(self, channels: List[int], embed_dim: int) -> None:
        super(Decoder, self).__init__()

        in_channs, out_channs = channels[:-1], channels[1:]
        self.upconv_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=in_chann,
                    out_channels=out_chann,
                    kernel_size=2,
                    stride=2,
                )
                for in_chann, out_chann in zip(in_channs, out_channs)
            ]
        )
        self.deconv_layers = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_chann, out_channels=out_chann, embed_dim=embed_dim
                )
                for in_chann, out_chann in zip(in_channs, out_channs)
            ]
        )
        self.final = nn.Conv2d(
            in_channels=channels[-1], out_channels=3, kernel_size=3, padding=1
        )

    def forward(
        self,
        input: torch.Tensor,
        skip_connections: List[torch.Tensor],
        time_embed: torch.Tensor,
    ) -> torch.Tensor:
        output = input
        for idx, (upconv, deconv) in enumerate(
            zip(self.upconv_layers, self.deconv_layers)
        ):
            output = upconv(output)
            output = torch.cat([output, skip_connections[idx]], dim=1)
            output = deconv(output, time_embed)

        output = self.final(output)
        return output
