from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

l1 = nn.L1Loss()
mse = nn.MSELoss()


def train_epoch(
    generators: List[nn.Module],
    discriminators: List[nn.Module],
    train_loader: DataLoader[torch.Tensor],
    optimizers: List[torch.optim.Optimizer],
    desc: str,
    lmbd_cycle: float = 10.0,
    lmbd_identity: float = 5.0,
) -> Tuple[float, float]:
    gen_v, gen_p = tuple(generators)
    disc_v, disc_p = tuple(discriminators)

    gen_v.train()
    gen_p.train()
    disc_v.train()
    disc_p.train()

    optim_g, optim_d = tuple(optimizers)

    gen_loss, disc_loss = 0.0, 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=desc)
    for batch, (Xr_v, Xr_p) in enumerate(pbar, start=1):
        Xr_v = Xr_v.to(DEVICE)
        Xr_p = Xr_p.to(DEVICE)

        # training both discriminators
        Xf_v = gen_v(Xr_p)  # fake vangogh from photo
        Xf_p = gen_p(Xr_v)  # fake photo from vangogh

        Dr_v, Df_v = disc_v(Xr_v), disc_v(Xf_v.detach())
        Dr_p, Df_p = disc_p(Xr_p), disc_p(Xf_p.detach())

        loss_dv = mse(Dr_v, torch.ones_like(Dr_v)) + mse(Df_v, torch.zeros_like(Df_v))
        loss_dp = mse(Dr_p, torch.ones_like(Dr_p)) + mse(Df_p, torch.zeros_like(Df_p))

        loss_d = (loss_dv + loss_dp) / 2

        optim_d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # training both generators
        # adversarial loss
        Df_v, Df_p = disc_v(Xf_v), disc_p(Xf_p)
        loss_a = mse(Df_v, torch.ones_like(Df_v)) + mse(Df_p, torch.ones_like(Df_p))

        # cycle consistency loss
        C_v, C_p = gen_v(Xf_p), gen_p(Xf_v)
        loss_c = l1(C_v, Xr_v) + l1(C_p, Xr_p)

        # identity loss
        loss_i = 0.0
        if lmbd_identity > 0:
            I_v, I_p = gen_v(Xr_v), gen_p(Xr_p)
            loss_i = l1(I_v, Xr_v) + l1(I_p, Xr_p)

        loss_g = loss_a + lmbd_cycle * loss_c + lmbd_identity * loss_i

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        gen_loss += loss_g.item()
        disc_loss += loss_d.item()

        pbar.set_postfix(
            {
                "Generator Loss": f"{gen_loss / batch:.4f}",
                "Discriminator Loss": f"{disc_loss / batch:.4f}",
            }
        )

    pbar.close()
    return gen_loss / num_batches, disc_loss / num_batches
