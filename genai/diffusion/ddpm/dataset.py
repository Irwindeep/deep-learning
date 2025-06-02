import os
from typing import Any, Callable, Optional
from torch.utils.data import Dataset
from PIL import Image


# kaggle dataset: https://www.kaggle.com/datasets/splcher/animefacedataset
class AnimeFaceDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable]) -> None:
        self.root = root
        self.transform = transform

        self.files = os.listdir(self.root)
        self.files = [os.path.join(self.root, file) for file in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Any:
        image_path = self.files[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
