from dl.nn.module import Module
from dl.tensor import Tensor

class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()

        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def add(self, *args: Module) -> None:
        idx_offset = 0
        for idx_str, _ in self._modules.items(): idx_offset = int(idx_str) + 1

        for idx, module in enumerate(args):
            self._modules[str(idx + idx_offset)] = module

    def forward(self, input: Tensor) -> Tensor:
        for _, module in self._modules.items():
            input = module(input)

        return input
