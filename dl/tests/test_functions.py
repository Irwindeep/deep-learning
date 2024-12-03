import unittest
import numpy as np
from dl.tensor import Tensor
from dl.functions import exp, sigmoid, tanh, relu

class TestFunctions(unittest.TestCase):
    def test_exp(self):
        t1 = Tensor([-1, 0, 1, 2], requires_grad=True)
        t2 = exp(t1)

        np.testing.assert_array_almost_equal(t2.data, np.exp(t1.data))

        t2.backward(Tensor(np.ones_like(t2.data)))
        np.testing.assert_array_almost_equal(t1.grad.data, np.ones_like(t2.data) * t2.data)

    def test_sigmoid(self):
        t1 = Tensor([-1, 0, 1, 2], requires_grad=True)
        t2 = sigmoid(t1)

        np.testing.assert_array_almost_equal(t2.data, 1/(1+np.exp(-t1.data)))

        t2.backward(Tensor(np.ones_like(t2.data)))
        np.testing.assert_array_almost_equal(t1.grad.data, np.ones_like(t2.data) * t2.data * (1-t2.data))
    
    def test_tanh(self):
        t1 = Tensor([-1, 0, 1, 2], requires_grad=True)
        t2 = tanh(t1)

        np.testing.assert_array_almost_equal(t2.data, np.tanh(t1.data))

        t2.backward(Tensor(np.ones_like(t2.data)))
        np.testing.assert_array_almost_equal(t1.grad.data, np.ones_like(t2.data) * (1-t2.data**2))

    def test_relu(self):
        t1 = Tensor([-1, 0, 1, 2], requires_grad=True)
        t2 = relu(t1)

        np.testing.assert_array_almost_equal(t2.data, [0, 0, 1, 2])

        t2.backward(Tensor(np.ones_like(t2.data)))
        np.testing.assert_array_almost_equal(t1.grad.data, [0, 0, 1, 1])
