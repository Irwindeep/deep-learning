import os
from typing import Any, Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


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
