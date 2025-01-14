from typing import Optional
from dl.tensor import Tensor
from dl.variable import Variable
from dl.nn.module import Module
import dl.functions as F

__all__ = [
    "AvgPool2d",
    "Conv2d",
    "Linear",
    "MaxPool2d"
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

class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weights = Variable(
            self.kernel_size, self.kernel_size,
            self.in_channels, self.out_channels
        )

        self.stride = stride
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        if self.padding > 0: input = F.pad(input, self.padding)
        return F.conv2d(input, self.weights, self.stride)
    
    def __repr__(self) -> str:
        return f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size})"

class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = 1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool_2d(input, self.kernel_size, self.stride)
    
    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"

class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = 1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(input, self.kernel_size, self.stride)
    
    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"

class Flatten(Module):
    def __init__(
        self,
        start_dim: int = 1,
        end_dim: int = -1
    ):
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        if self.end_dim < 0:
            self.end_dim = input.data.ndim + self.end_dim

        new_shape = input.shape[:self.start_dim] + (-1,) + input.shape[self.end_dim + 1:]
        return input.reshape(*new_shape)
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"

class RNNCell(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation: str = "tanh"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.activation = activation

        self.weight_ih = Variable(self.hidden_size, self.input_size)
        self.weight_hh = Variable(self.hidden_size, self.hidden_size)

        self.bias_ih: Optional[Tensor] = None
        self.bias_hh: Optional[Tensor] = None
        if bias:
            self.bias_ih = Variable(self.hidden_size)
            self.bias_hh = Variable(self.hidden_size)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        return F.rnn_cell(
            input, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, hx
        )
    
    def __repr__(self):
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})"
