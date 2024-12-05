import dl.functions as F
from dl.tensor import Tensor
from dl.nn.module import Module

class MSELoss(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
