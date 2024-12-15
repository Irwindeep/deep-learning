from typing import List, Optional, Tuple
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

def pad(tensor: Tensor, padding: int) -> Tensor:
    pad_width = ((0, 0), (padding, padding), (padding, padding), (0, 0))

    data = np.pad(tensor.data, pad_width, mode="constant", constant_values=0)
    requires_grad = tensor.requires_grad
    
    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad[padding:-padding, :][:, padding:-padding]

        depends_on.append(Dependency(tensor, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def conv_im2col(input: NDArray, weights: NDArray, stride: int) -> Tuple[NDArray, NDArray]:
    batch_size, h, w, in_c = input.shape
    k, _, _, out_c = weights.shape

    batch_stride, h_s, w_s, c_s = input.strides
    out_h = (h - k)//stride + 1
    out_w = (w - k)//stride + 1
    
    inner_dim = k * k * in_c
    input_patches = np.lib.stride_tricks.as_strided(
        input, shape=(batch_size, out_h, out_w, k, k, in_c),
        strides=(batch_stride, h_s*stride, w_s*stride, h_s, w_s, c_s)
    ).reshape(-1, inner_dim)

    data = input_patches @ weights.reshape(-1, out_c)
    return input_patches, data.reshape(batch_size, out_h, out_w, out_c)

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

def softmax(tensor: Tensor) -> Tensor:
    output = exp(tensor - np.max(tensor.data, axis=-1, keepdims=True))
    output = output/np.sum(output.data, axis=-1, keepdims=True)

    return output

# Neural Network
def linear(input: Tensor, weights: Tensor, bias: Optional[Tensor]) -> Tensor:
    x = input @ weights
    if bias is not None: x = x + bias

    return x

def conv2d(input: Tensor, weights: Tensor, stride: int) -> Tensor:
    _, h, w, in_c = input.data.shape
    k, _, _, out_c = weights.data.shape
    out_h, out_w = (h - k)//stride + 1, (w - k)//stride + 1

    input_patches, data = conv_im2col(input.data, weights.data, stride)
    requires_grad = input.requires_grad or weights.requires_grad

    depends_on: List[Dependency] = []
    if input.requires_grad:
        def grad_fn1(grad: NDArray) -> NDArray:
            pad_h = (h - 1) * stride + k - out_h
            pad_w = (w - 1) * stride + k - out_w
            pad_top, pad_left = pad_h//2, pad_w//2
            pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left

            return conv_im2col(
                np.pad(
                    grad,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                    constant_values=0
                ),
                np.flip(weights.data, axis=(0, 1)).reshape(k, k, out_c, in_c),
                stride
            )[1]

        depends_on.append(Dependency(input, grad_fn1))

    if weights.requires_grad:
        def grad_fn2(grad: NDArray) -> NDArray:
            grad = grad.reshape(-1, out_c)
            return (input_patches.T @ grad).reshape(k, k, in_c, out_c)

        depends_on.append(Dependency(weights, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def max_pool_2d(input: Tensor, kernel_size: int, stride: int) -> Tensor:
    batch_size, h, w, c = input.data.shape
    out_h, out_w = (h - kernel_size)//stride + 1, (w - kernel_size)//stride + 1

    batch_stride, h_s, w_s, c_s = input.data.strides
    strided = np.lib.stride_tricks.as_strided(
        input.data,
        shape=(batch_size, out_h, out_w, kernel_size, kernel_size, c),
        strides=(batch_stride, h_s*stride, w_s*stride, h_s, w_s, c_s)
    )

    data = np.max(strided, axis=(3, 4))
    requires_grad = input.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            grad_input = np.zeros_like(input.data)
            max_mask = (strided == np.max(strided, axis=(3, 4), keepdims=True))

            grad_broadcast = grad[:, :, :, None, None, :]
            grad_strided = max_mask * grad_broadcast

            for i in range(kernel_size):
                for j in range(kernel_size):
                    grad_input[:, i::stride, j::stride, :] += grad_strided[:, :, :, i, j, :]
            
            return grad_input

        depends_on.append(Dependency(input, grad_fn))

    return Tensor(data, requires_grad, depends_on)

def avg_pool2d(input: Tensor, kernel_size: int, stride: int) -> Tensor:
    weights = Tensor(np.ones((kernel_size, kernel_size, input.shape[0], input.shape[0])))

    return conv2d(input, weights, stride)

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

def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    input_data: NDArray = np.exp(input.data - np.max(input.data, axis=-1, keepdims=True))
    input_data = input_data/np.sum(input_data, axis=-1, keepdims=True)

    data = - np.sum(target.data * np.log(input_data + 1e-12))/input_data.shape[0]
    requires_grad = input.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: NDArray) -> NDArray:
            return grad * (input_data - target.data)
        
        depends_on.append(Dependency(input, grad_fn))

    return Tensor(data, requires_grad, depends_on)
