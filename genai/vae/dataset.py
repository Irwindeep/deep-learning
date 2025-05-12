import os
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


# kaggle dataset - https://www.kaggle.com/datasets/splcher/animefacedataset
class AnimeFaceDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]) -> None:
        super(AnimeFaceDataset, self).__init__()

        self.root = root
        self.image_files = os.listdir(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> Any:
        image_file = self.image_files[index]
        image = Image.open(os.path.join(self.root, image_file))

        if self.transform:
            return self.transform(image)

        return image


def get_transform(size: int) -> Tuple[Callable, Callable]:
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    inv_norm = transforms.Normalize((-1, -1, -1), (2, 2, 2))

    return transform, inv_norm


def get_datasets(root: str, transform: Callable, seed: int = 12) -> Tuple[Dataset, ...]:
    torch.manual_seed(seed)

    dataset = AnimeFaceDataset(root=root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def get_loaders(
    *datasets: Dataset, batch_size: int
) -> Tuple[DataLoader[torch.Tensor], ...]:
    train_dataset, val_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader
