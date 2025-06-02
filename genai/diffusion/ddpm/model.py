from typing import List
from .utils import Encoder, Decoder
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self, encoder_channels: List[int], bottleneck_size: int, embed_dim: int
    ) -> None:
        super(UNet, self).__init__()

        self.embed_dim = embed_dim
        self.time_proj = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
        )

        self.encoder = Encoder(
            channels=encoder_channels,
            bottleneck_size=bottleneck_size,
            embed_dim=embed_dim,
        )
        self.decoder = Decoder(
            channels=[bottleneck_size] + encoder_channels[1:][::-1], embed_dim=embed_dim
        )

    def forward(self, input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embed = self._timestep_embd(timestep)
        time_embed = self.time_proj(time_embed)

        bottleneck, skip_conns = self.encoder(input, time_embed)
        output = self.decoder(bottleneck, skip_conns[::-1], time_embed)
        return output

    def _timestep_embd(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        embd = torch.log(torch.tensor(1e4)) / (half_dim - 1)

        embd = torch.exp(-embd * torch.arange(half_dim, dtype=torch.float32)).to(
            timestep.device
        )
        embd = (
            timestep.float()[:, None] * embd[None, :]
        )  # shape: (batch_size, half_dim)

        embd = torch.cat([torch.sin(embd), torch.cos(embd)], dim=1)
        return embd
