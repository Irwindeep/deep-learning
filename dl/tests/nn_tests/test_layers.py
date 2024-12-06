import unittest
import numpy as np
from dl.tensor import Tensor
from dl.nn import Linear

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
