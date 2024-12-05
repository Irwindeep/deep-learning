from dl.nn.module import Module
from dl.tensor import Tensor

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()

        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, input: Tensor) -> Tensor:
        for _, module in self._modules.items():
            input = module(input)

        return input
