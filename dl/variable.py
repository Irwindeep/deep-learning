import numpy as np
from dl.tensor import Tensor

class Variable(Tensor):
    def __init__(self, *shape, random_seed: int = 20) -> None:
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)

        self.u = np.zeros_like(data)
        self.v = np.zeros_like(data)
