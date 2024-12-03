from typing import List, NamedTuple, Callable, Optional, Union
from numpy.typing import NDArray
import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[NDArray], NDArray]

Arrayable = Union[float, List, NDArray]
Tensorable = Union['Tensor', float, NDArray]

def ensure_array(arrayable: Arrayable) -> NDArray:
    if isinstance(arrayable, np.ndarray): return arrayable
    return np.array(arrayable, dtype=np.float64)

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor): return tensorable
    return Tensor(tensorable)

class Tensor:
    def __init__(
            self,
            data: Arrayable,
            requires_grad: bool = False,
            depends_on: List[Dependency] = []
    ) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on

        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad: self.zero_grad()

    @property
    def data(self) -> NDArray:
        return self._data
    
    @data.setter
    def data(self, new_data: Arrayable) -> None:
        self._data = ensure_array(new_data)
        self.grad = None

    @property
    def T(self) -> 'Tensor':
        return _transpose(self)

    def sum(self) -> 'Tensor':
        data = np.array(self.data.sum())
        requires_grad = self.requires_grad

        depends_on: List[Dependency] = []
        if requires_grad:
            def grad_fn(grad: NDArray) -> NDArray:
                return grad * np.ones_like(self.data)
            
            depends_on.append(Dependency(self, grad_fn))

        return Tensor(data, requires_grad, depends_on)
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
    
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
    
    def __add__(self, other: Tensorable) -> 'Tensor':
        return _add(self, ensure_tensor(other))
    
    def __radd__(self, other: Tensorable) -> 'Tensor':
        return _add(ensure_tensor(other), self)
    
    def __iadd__(self, other: Tensorable) -> 'Tensor':
        self.data += ensure_tensor(other).data
        return self

    def __neg__(self) -> 'Tensor':
        return _neg(self)
    
    def __sub__(self, other: Tensorable) -> 'Tensor':
        return _sub(self, ensure_tensor(other))
    
    def __rsub__(self, other: Tensorable) -> 'Tensor':
        return _sub(ensure_tensor(other), self)
    
    def __isub__(self, other: Tensorable) -> 'Tensor':
        self.data -= ensure_tensor(other).data
        return self
    
    def __mul__(self, other: Tensorable) -> 'Tensor':
        return _mul(self, ensure_tensor(other))
    
    def __rmul__(self, other: Tensorable) -> 'Tensor':
        return _mul(ensure_tensor(other), self)
    
    def __imul__(self, other: Tensorable) -> 'Tensor':
        self.data *= ensure_tensor(other).data
        return self
    
    def __matmul__(self, other: Tensorable) -> 'Tensor':
        return _matmul(self, ensure_tensor(other))
    
    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: NDArray) -> NDArray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added): grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1: grad = grad.sum(axis=i, keepdims=True)
            
            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: NDArray) -> NDArray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added): grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1: grad = grad.sum(axis=i, keepdims=True)
            
            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: NDArray) -> NDArray:
            grad = grad * t2.data

            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added): grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1: grad = grad.sum(axis=i, keepdims=True)
            
            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: NDArray) -> NDArray:
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added): grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1: grad = grad.sum(axis=i, keepdims=True)
            
            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    
    depends_on: List[Dependency] = []
    if requires_grad: depends_on.append(Dependency(t, lambda x: -x))

    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + (-t2)

def _transpose(t: Tensor, axes: Optional[tuple[int, int]] = None) -> Tensor:
    data = t.data.T if axes is None else np.swapaxes(t.data, *axes)
    requires_grad = t.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad.T if axes is None else np.swapaxes(grad, *axes)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: NDArray) -> NDArray:
            grad = grad @ t2.data.T            
            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: NDArray) -> NDArray:
            grad = t1.data.T @ grad            
            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _slice(t: Tensor, idxs: slice) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad

            return bigger_grad

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data, requires_grad, depends_on)
