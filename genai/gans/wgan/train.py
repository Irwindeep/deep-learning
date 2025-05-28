from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import gradient_penalty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    gen: nn.Module,
    critic: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    optim_g: torch.optim.Optimizer,
    optim_c: torch.optim.Optimizer,
    desc: str,
    n_critic: int = 5,
    noise_dim: int = 100,
    lmbd_gp: float = 10.0,
) -> Tuple[float, float]:
    gen.train()
    critic.train()

    gen_loss, critic_loss = 0.0, 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=desc)
    for batch, X_r in enumerate(pbar, start=1):
        X_r = X_r.to(DEVICE)

        c_loss = 0.0
        for _ in range(n_critic):
            noise = torch.randn(X_r.shape[0], noise_dim).to(DEVICE)
            X_g = gen(noise)

            critic_r = critic(X_r)
            critic_g = critic(X_g)

            grad_penalty = gradient_penalty(critic, X_r, X_g, device=DEVICE)
            loss_c = (
                torch.mean(critic_g) - torch.mean(critic_r)
            ) + lmbd_gp * grad_penalty

            optim_c.zero_grad()
            loss_c.backward()
            optim_c.step()

            c_loss += loss_c.item()

        noise = torch.randn(X_r.shape[0], noise_dim).to(DEVICE)
        X_g = gen(noise)

        critic_g = critic(X_g)
        loss_g = -torch.mean(critic_g)

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        gen_loss += loss_g.item()
        critic_loss += c_loss / n_critic

        pbar.set_postfix(
            {
                "Generator Loss": f"{gen_loss/batch:.4f}",
                "Critic Loss": f"{critic_loss/batch:.4f}",
            }
        )

    pbar.close()
    return gen_loss / num_batches, critic_loss / num_batches
