from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CelebA
from torchvision.transforms import transforms


def get_transform() -> Tuple[Callable, Callable]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inv_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2))

    return transform, inv_norm


def get_datasets(transform: Callable, seed: int = 12) -> Tuple[Dataset, ...]:
    torch.manual_seed(seed)

    train_dataset = CelebA(
        root="celeba", split="train", transform=transform, download=True
    )
    test_dataset = CelebA(
        root="celeba", split="test", transform=transform, download=False
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_loaders(
    *datasets: Dataset, batch_size: int
) -> Tuple[DataLoader[torch.Tensor], ...]:
    train_dataset, val_dataset, test_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
