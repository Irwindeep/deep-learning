from typing import List
import numpy as np
from numpy.typing import NDArray
from dl.tensor import Tensor, Dependency

def exp(tensor: Tensor) -> Tensor:
    data = np.exp(tensor.data)
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * data
        
        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def sigmoid(tensor: Tensor) -> Tensor:
    data = 1/(1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * data * (1 - data)
        
        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * (1 - data*data)

        depends_on.append(Dependency(tensor, grad_fn))
        
    return Tensor(data, requires_grad, depends_on)

def relu(tensor: Tensor) -> Tensor:
    data = np.maximum(0, tensor.data)
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * np.where(data > 0, 1, 0)

        depends_on.append(Dependency(tensor, grad_fn))
        
    return Tensor(data, requires_grad, depends_on)
