import unittest
import numpy as np
from dl.tensor import Tensor

class TestTensorTranspose(unittest.TestCase):
    def test_simple_transpose(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        t2 = t1.T

        assert t2.data.tolist() == [[1, 4], [2, 5], [3, 6]]
        
        t2.backward(Tensor([[1, 1], [1, 1], [1, 1]]))
        np.testing.assert_array_equal(t1.grad.data, np.ones_like(t1.data))
