from typing import Tuple
import numpy as np
from dl.tensor import Tensor

class Variable(Tensor):
    def __init__(self, *shape, random_seed: int = 20, n_axis: Tuple[int, ...] = ()) -> None:
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        data = np.random.randn(*shape)
        if n_axis != (): data = data/data.sum(axis=n_axis)

        super().__init__(data, requires_grad=True)

        self.u = np.zeros_like(data)
        self.v = np.zeros_like(data)
