import dl.functions as F
from dl.tensor import Tensor
from dl.nn.module import Module

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)
    
class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)
