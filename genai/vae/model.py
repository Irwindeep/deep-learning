from typing import List, Tuple

import torch
import torch.nn as nn

from .utils import ConvBlock, UpsampleBlock


class VAE(nn.Module):
    def __init__(self, encoder_channels: List[int], kernel_size: int) -> None:
        super(VAE, self).__init__()
        in_channs, out_channs = encoder_channels[:-1], encoder_channels[1:]

        self.encoder = nn.Sequential(
            *[
                self._make_layer(in_chann, out_chann, kernel_size=kernel_size)
                for in_chann, out_chann in list(zip(in_channs, out_channs))[:-1]
            ]
        )

        self.mu_head = self._make_layer(
            in_channs[-1], out_channs[-1], kernel_size=kernel_size
        )
        self.logvar_head = self._make_layer(
            in_channs[-1], out_channs[-1], kernel_size=kernel_size
        )

        self.decoder = nn.Sequential(
            *[
                UpsampleBlock(
                    out_chann,
                    in_chann,
                    kernel_size=kernel_size,
                    stride=2,
                    only_conv=(in_chann == in_channs[0]),
                )
                for out_chann, in_chann in list(zip(out_channs, in_channs))[::-1]
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        encoded_ip = self.encoder(input)

        mu = self.mu_head(encoded_ip)
        logvar = self.logvar_head(encoded_ip)

        return mu, logvar

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        output = self.decoder(input)
        return self.sigmoid(output)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mu, logvar = self.encode(input)

        # reparametrization trick
        eps = torch.rand_like(logvar, device=logvar.device)
        decoder_input = mu + eps * logvar

        output = self.decode(decoder_input)
        return output, mu, logvar

    def _make_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Sequential(ConvBlock(*args, **kwargs), nn.MaxPool2d(2, 2))
