import unittest
import numpy as np
from dl.tensor import Tensor
from dl.nn import (
    Linear,
    Conv2d,
    MaxPool2d,
    Flatten
)
import torch

class TestLayers(unittest.TestCase):
    def test_linear(self):
        t1 = Tensor([[1, 2, 3, 4]], requires_grad=True)
        linear = Linear(in_features=4, out_features=10)

        t2 = linear(t1)

        np.testing.assert_array_almost_equal(
            t2.data,
            (t1 @ linear.weights + linear.bias).data
        )

        t2.backward(Tensor(np.ones((1, 10))))

        np.testing.assert_array_almost_equal(
            t1.grad.data,
            (np.ones((1, 10)) @ linear.weights.T.data)
        )

    def test_conv2d(self):
        def conv_reference(Z, weight):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            W_torch = torch.tensor(weight).permute(3,2,0,1)
            
            out = torch.nn.functional.conv2d(Z_torch, W_torch)
            return out.permute(0,2,3,1).contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        
        conv_2d = Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        weight = conv_2d.weights.data

        output_torch = conv_reference(t1_np, weight)
        output_dl = conv_2d(t1)

        assert output_torch.shape == output_dl.shape
        np.testing.assert_array_almost_equal(output_dl.data, output_torch)

    def test_conv2d_backprop(self):
        def backprop_reference(Z, weight):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            W_torch = torch.tensor(weight).permute(3,2,0,1)
            Z_torch.requires_grad, W_torch.requires_grad = True, True
            
            out = torch.nn.functional.conv2d(Z_torch, W_torch, padding=1)
            out.backward(torch.tensor(np.ones(out.shape)))

            return Z_torch.grad.permute(0,2,3,1).contiguous().numpy(), W_torch.grad.permute(2, 3, 1, 0).contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        
        conv_2d = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        weight = conv_2d.weights.data

        input_grad_torch, weight_grad_torch = backprop_reference(t1_np, weight)

        output_dl = conv_2d(t1)
        output_dl.backward(Tensor(np.ones(output_dl.shape)))

        assert t1.grad.data.shape == input_grad_torch.shape
        assert conv_2d.weights.grad.shape == weight_grad_torch.shape
        np.testing.assert_array_almost_equal(t1.grad.data, input_grad_torch)
        np.testing.assert_array_almost_equal(conv_2d.weights.grad.data, weight_grad_torch)

    def test_maxpool2d(self):
        def max_pool_reference(Z, kernel_size, stride):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            out = torch.nn.functional.max_pool2d(Z_torch, kernel_size, stride)

            return out.permute(0,2,3,1).contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        
        max_pool_2d = MaxPool2d(kernel_size=3, stride=3)

        output_torch = max_pool_reference(t1_np, 3, 3)
        output_dl = max_pool_2d(t1)

        assert output_torch.shape == output_dl.shape
        np.testing.assert_array_almost_equal(output_dl.data, output_torch)

    def test_max_pool_backprop(self):
        def backprop_reference(Z, kernel_size, stride):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            Z_torch.requires_grad = True
            out = torch.nn.functional.max_pool2d(Z_torch, kernel_size, stride)

            out.backward(torch.tensor(np.ones(out.shape)))
            return Z_torch.grad.permute(0,2,3,1).contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        
        max_pool_2d = MaxPool2d(kernel_size=3, stride=3)

        input_grad_torch = backprop_reference(t1_np, 3, 3)

        output_dl = max_pool_2d(t1)
        output_dl.backward(Tensor(np.ones(output_dl.shape)))

        assert t1.grad.data.shape == input_grad_torch.shape
        np.testing.assert_array_almost_equal(t1.grad.data, input_grad_torch)

    def test_flatten(self):
        def flatten_reference(Z):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            out = torch.nn.Flatten()(Z_torch)

            return out.contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        flatten = Flatten()
        
        t2 = flatten(t1)
        t3 = flatten(t1_np)

        assert t2.shape == (10, 243)
        assert t3.shape == (10, 243)
        np.testing.assert_array_almost_equal(t2.data, t3)
    
    def test_flatten_backprop(self):
        def backprop_reference(Z):
            Z_torch = torch.tensor(Z).permute(0,3,1,2)
            Z_torch.requires_grad = True
            out = torch.nn.Flatten()(Z_torch)

            out.backward(torch.tensor(np.ones(out.shape)))
            return Z_torch.grad.permute(0,2,3,1).contiguous().numpy()
        
        t1_np = np.random.randn(10, 9, 9, 3)
        t1 = Tensor(t1_np, requires_grad=True)
        flatten = Flatten()
        
        t2 = flatten(t1)
        t2.backward(Tensor(np.ones(t2.shape)))

        input_grad_torch = backprop_reference(t1_np)

        assert t1.grad.data.shape == input_grad_torch.shape
        assert t1.grad.data.shape == (10, 9, 9, 3)
        np.testing.assert_array_almost_equal(t1.grad.data, input_grad_torch)
