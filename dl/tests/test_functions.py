import unittest
import numpy as np
import dl
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

    def test_tensor_split(self):
        t = Tensor(np.random.randn(12, 4), requires_grad=True)
        splitted_tensors = dl.split(t, sections=4, axis=0)

        assert all(_t.data.shape == (3, 4) for _t in splitted_tensors)

        splitted_tensors[0].backward(Tensor(np.ones((3, 4))))
        np.testing.assert_array_equal(t.grad.data[0:3], np.ones((3, 4)))
        np.testing.assert_array_equal(t.grad.data[3:12], np.zeros((9, 4)))

        splitted_tensors[1].backward(Tensor(np.ones((3, 4))))
        np.testing.assert_array_equal(t.grad.data[0:6], np.ones((6, 4)))
        np.testing.assert_array_equal(t.grad.data[6:12], np.zeros((6, 4)))

        splitted_tensors[2].backward(Tensor(np.ones((3, 4))))
        np.testing.assert_array_equal(t.grad.data[0:9], np.ones((9, 4)))
        np.testing.assert_array_equal(t.grad.data[9:12], np.zeros((3, 4)))

        splitted_tensors[3].backward(Tensor(np.ones((3, 4))))
        np.testing.assert_array_equal(t.grad.data, np.ones((12, 4)))
