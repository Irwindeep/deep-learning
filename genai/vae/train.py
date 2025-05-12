import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def val_epoch(
    model: nn.Module, val_loader: DataLoader[torch.Tensor], loss_fn: nn.Module
) -> float:
    model.eval()

    val_loss, num_batches = 0.0, len(val_loader)
    with torch.no_grad():
        for X in val_loader:
            X = X.to(device)
            X_recon, mu, logvar = model(X)

            loss = loss_fn(X_recon, X, mu, logvar)
            val_loss += loss.item()

    return val_loss / num_batches


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    desc: str,
) -> float:
    model.train()

    train_loss, num_batches = 0.0, len(train_loader)
    pbar = tqdm(train_loader, desc=desc)

    for batch, X in enumerate(pbar, start=1):
        optim.zero_grad()
        X = X.to(device)
        X_recon, mu, logvar = model(X)

        loss = loss_fn(X_recon, X, mu, logvar)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item()}:.4f",
                "Train Loss": f"{train_loss/batch:.4f}",
            }
        )

    pbar.close()
    return train_loss / num_batches
