import unittest
import numpy as np
from dl.tensor import Tensor
from dl.nn import Sequential, Linear, ReLU

class TestContainers(unittest.TestCase):
    def test_sequential(self):
        t1 = Tensor([[1, -2, 3, 0]], requires_grad=True)

        model = Sequential(
            Linear(in_features=4, out_features=10),
            ReLU(),
            Linear(in_features=10, out_features=1)
        )
        t2 = model(t1)

        np.testing.assert_array_almost_equal(
            t2.data,
            model._modules['2'](model._modules['1'](model._modules['0'](t1))).data
        )

        t2.backward(Tensor(np.ones((1, 1))))
        assert np.all(t1.grad.data != 0)
