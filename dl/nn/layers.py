from typing import Optional
from dl.tensor import Tensor
from dl.variable import Variable
from dl.nn.module import Module
import dl.functions as F

__all__ = [
    "Linear"
]

class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = Variable(self.in_features, self.out_features)
        self.bias: Optional[Tensor] = None
        if bias:
            self.bias = Variable(self.out_features)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weights, self.bias)
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={True if self.bias else False})"
