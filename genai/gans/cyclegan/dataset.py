import os
from PIL import Image
from typing import Callable, Literal, Optional, Tuple, Any
from torch.utils.data import Dataset


# kaggle dataset: https://www.kaggle.com/datasets/def0017/vangogh2photo/data
class VanGogh2Photo(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform

        self.file_dir = split if split == "train" else "test"
        self.vangogh_files = os.listdir(os.path.join(self.root, self.file_dir + "A"))
        self.photo_files = os.listdir(os.path.join(self.root, self.file_dir + "B"))

        self.dataset_len = max(len(self.vangogh_files), len(self.photo_files))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        vangogh_path = self.vangogh_files[idx % len(self.vangogh_files)]
        photo_path = self.photo_files[idx % len(self.photo_files)]

        vangogh_path = os.path.join(self.root, self.file_dir + "A", vangogh_path)
        photo_path = os.path.join(self.root, self.file_dir + "B", photo_path)

        vangogh = Image.open(vangogh_path)
        photo = Image.open(photo_path)

        if self.transform:
            vangogh = self.transform(vangogh)
            photo = self.transform(photo)

        return vangogh, photo
