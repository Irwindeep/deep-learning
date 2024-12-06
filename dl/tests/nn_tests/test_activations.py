import unittest
import numpy as np
from dl.tensor import Tensor
from dl.nn import ReLU, Sigmoid, Tanh

class TestActivations(unittest.TestCase):
    def test_relu(self):
        t1 = Tensor([1, -2, 3, 0], requires_grad=True)

        activation = ReLU()
        t2 = activation(t1)

        assert t2.data.tolist() == [1, 0, 3, 0]
        t2.backward(Tensor(np.ones(4)))

        assert t1.grad.data.tolist() == [1, 0, 1, 0]

    def test_sigmoid(self):
        t1 = Tensor([1, -2, 3, 0], requires_grad=True)

        activation = Sigmoid()
        t2 = activation(t1)

        np.testing.assert_array_almost_equal(t2.data, 1/(1+np.exp(-t1.data)))
        t2.backward(Tensor(np.ones(4)))

        np.testing.assert_array_almost_equal(t1.grad.data, t2.data * (1 - t2.data))

    def test_tanh(self):
        t1 = Tensor([1, -2, 3, 0], requires_grad=True)

        activation = Tanh()
        t2 = activation(t1)

        np.testing.assert_array_almost_equal(t2.data, np.tanh(t1.data))
        t2.backward(Tensor(np.ones(4)))

        np.testing.assert_array_almost_equal(t1.grad.data, (1 - t2.data * t2.data))
