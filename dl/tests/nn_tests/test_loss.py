import unittest
from dl.tensor import Tensor
from dl.nn import MSELoss

class TestLosses(unittest.TestCase):
    def test_mse_loss(self):
        t1 = Tensor([1, 2, 3, 4], requires_grad=True)
        t2 = Tensor([1, 1, 1, 1], requires_grad=True)

        loss = MSELoss()
        t3 = loss(t1, t2)

        assert t3.data == 3.5
        t3.backward()

        assert t1.grad.data.tolist() == [0, 0.5, 1, 1.5]
        assert t2.grad.data.tolist() == [0, -0.5, -1, -1.5]
