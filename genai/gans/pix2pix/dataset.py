import os
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset


class Label2Facade(Dataset):
    def __init__(
        self,
        root_labels: str,
        root_facades: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_labels = root_labels
        self.root_facades = root_facades
        self.transform = transform

        # since dataset is aligned, only one path is requires
        self.paths = os.listdir(self.root_labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path = self.paths[idx]
        label_path = os.path.join(self.root_labels, path)
        facade_path = os.path.join(self.root_facades, path)

        label = Image.open(label_path)
        facade = Image.open(facade_path)

        if self.transform:
            label = self.transform(label)
            facade = self.transform(facade)

        return label, facade
