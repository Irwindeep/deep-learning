import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    alpha: torch.Tensor,
    desc: str,
    timesteps: int,
    leave: bool = True,
) -> float:
    model.train()

    train_loss, num_batches = 0.0, len(train_loader)
    pbar = tqdm(train_loader, desc=desc, leave=leave)
    for batch, X in enumerate(pbar, start=1):
        X = X.to(DEVICE)

        timestep = torch.randint(0, timesteps, size=(X.size(0),), device=DEVICE)
        noise = torch.randn_like(X)

        # diffusion
        X_t = (
            torch.sqrt(alpha[timestep])[:, None, None, None] * X
            + torch.sqrt(1 - alpha[timestep])[:, None, None, None] * noise
        )
        pred = model(X_t, timestep)

        loss = loss_fn(pred, noise)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item():.4f}",
                "Train Loss": f"{train_loss / batch:.4f}",
            }
        )

    pbar.close()
    return train_loss / num_batches
