from typing import List, Optional
import numpy as np
from numpy.typing import NDArray
from dl.tensor import Tensor, Dependency

# Usefule Functions
def exp(tensor: Tensor) -> Tensor:
    data = np.exp(tensor.data)
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * data
        
        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def log(tensor: Tensor) -> Tensor:
    data = np.log(tensor.data + 1e-12)
    requires_grad = tensor.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * 1/(tensor.data + 1e-12)
        
        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

# Activation Functions
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

# Neural Network
def linear(input: Tensor, weights: Tensor, bias: Optional[Tensor]) -> Tensor:
    x = input @ weights
    if bias is not None: x = x + bias

    return x

# Losses
def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    errors = (input - target)
    loss = (errors**2).sum()/errors.shape[0]

    return loss

def bce_loss(input: Tensor, target: Tensor) -> Tensor:
    loss = - (
        target.T @ log(input) + 
        (1-target).T @ log(1 - input)
    ).sum()/input.shape[0]
    
    return loss
