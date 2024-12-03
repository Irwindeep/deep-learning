from typing import List, NamedTuple, Callable, Optional, Union
from numpy.typing import NDArray
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[NDArray], NDArray]

Arrayable = Union[float, List, NDArray]

def ensure_array(arrayable: Arrayable) -> NDArray:
    if isinstance(arrayable, np.ndarray): return arrayable
    return np.array(arrayable)

class Tensor:
    def __init__(
            self,
            data: Arrayable,
            requires_grad: bool = False,
            depends_on: List[Dependency] = []
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on

        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad: self.zero_grad()

    def sum(self) -> 'Tensor':
        data = np.array(self.data.sum())
        requires_grad = self.requires_grad

        if requires_grad:
            def grad_fn(grad: NDArray) -> NDArray:
                return grad * np.ones_like(self.data)
            
            depends_on = [Dependency(self, grad_fn)]

        else: depends_on = []

        return Tensor(data, requires_grad, depends_on)
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))
    
    def backward(self, grad: Optional['Tensor'] = None) -> None:
        assert self.grad, "Called backward on non-requires_grad tensor"

        if grad is None:
            if self.shape == (): grad = Tensor(1)
            else: raise RuntimeError("grad must be specified for non-zero tensor")
        
        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, requires_grad={self.requires_grad})"
