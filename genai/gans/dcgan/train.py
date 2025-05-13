from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    gen: nn.Module,
    disc: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    loss_fn: nn.Module,
    optim_gen: torch.optim.Optimizer,
    optim_disc: torch.optim.Optimizer,
    desc: str,
    noise_dim: int,
) -> Tuple[float, float]:
    gen.train()
    disc.train()

    gen_loss, disc_loss, num_batches = 0.0, 0.0, len(train_loader)
    pbar = tqdm(train_loader, desc=desc)

    for batch, X_r in enumerate(pbar, start=1):
        X_r = X_r.to(device)
        noise = torch.randn(X_r.size(0), noise_dim).to(device)

        X_g = gen(noise)

        # discriminator update
        disc_r = disc(X_r)
        disc_g = disc(X_g.detach())

        loss_disc = (
            loss_fn(disc_r, torch.ones_like(disc_r))
            + loss_fn(disc_g, torch.zeros_like(disc_g))
        ) / 2

        optim_disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()

        # generator update
        disc_g = disc(X_g)
        loss_gen = loss_fn(disc_g, torch.ones_like(disc_g))

        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        disc_loss += loss_disc.item()
        gen_loss += loss_gen.item()
        pbar.set_postfix(
            {
                "Disc Loss": f"{disc_loss/batch:.4f}",
                "Gen Loss": f"{gen_loss/batch:.4f}",
            }
        )

    pbar.close()
    return gen_loss / num_batches, disc_loss / num_batches
