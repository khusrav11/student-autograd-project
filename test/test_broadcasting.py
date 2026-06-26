import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import unittest
from src.tensor import Tensor

def allclose(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)

class TestBroadcastingBackward(unittest.TestCase):

    def test_bias_add_grad_shape(self):
        # (64,3) + (3,): bias grad should be (3,) not (64,3)
        X = Tensor(np.random.randn(64, 3), requires_grad=True)
        b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        (X + b).sum().backward()
        self.assertEqual(b.grad.shape, (3,))
        self.assertEqual(X.grad.shape, (64, 3))

    def test_bias_add_grad_values(self):
        # bias grad = batch_size (each bias element touched by every row)
        batch = 10
        X = Tensor(np.zeros((batch, 3)), requires_grad=True)
        b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        (X + b).sum().backward()
        self.assertTrue(allclose(b.grad, np.full(3, float(batch))))

    def test_outer_product_shapes(self):
        # (5,1) * (1,4): grads should keep original shapes
        a = Tensor(np.random.randn(5, 1), requires_grad=True)
        b = Tensor(np.random.randn(1, 4), requires_grad=True)
        (a * b).sum().backward()
        self.assertEqual(a.grad.shape, (5, 1))
        self.assertEqual(b.grad.shape, (1, 4))

    def test_outer_product_values(self):
        # L = sum(a * b): dL/da_i = sum(b), dL/db_j = sum(a)
        a_data = np.arange(1, 6, dtype=float).reshape(5, 1)
        b_data = np.arange(1, 5, dtype=float).reshape(1, 4)
        a = Tensor(a_data, requires_grad=True)
        b = Tensor(b_data, requires_grad=True)
        (a * b).sum().backward()
        self.assertTrue(allclose(a.grad, np.full((5, 1), np.sum(b_data))))
        self.assertTrue(allclose(b.grad, np.full((1, 4), np.sum(a_data))))

    def test_scalar_add_to_matrix(self):
        # scalar grad = sum of all output grads
        s = Tensor(2.0, requires_grad=True)
        M = Tensor(np.ones((3, 4)), requires_grad=True)
        (s + M).sum().backward()
        self.assertTrue(allclose(s.grad, 12.0))

    def test_scalar_mul_by_matrix(self):
        s = Tensor(3.0, requires_grad=True)
        M_data = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        M = Tensor(M_data, requires_grad=True)
        (s * M).sum().backward()
        self.assertTrue(allclose(s.grad, np.sum(M_data)))
        self.assertTrue(allclose(M.grad, 3.0 * np.ones((2, 3))))

    def test_broadcast_subtraction(self):
        # (4,3) - (3,): bias grad = -batch_size
        X = Tensor(np.ones((4, 3)), requires_grad=True)
        b = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        (X - b).sum().backward()
        self.assertEqual(b.grad.shape, (3,))
        self.assertTrue(allclose(b.grad, np.full(3, -4.0)))

    def test_keepdims_sum_grad(self):
        x = Tensor(np.array([[1.0, 2.0],[3.0, 4.0]]), requires_grad=True)
        x.sum(axis=1, keepdims=True).sum().backward()
        self.assertTrue(allclose(x.grad, np.ones((2, 2))))

    def test_row_vector_broadcast(self):
        # (1,4) added to (3,4): row vector grad = 3 per element
        r = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]), requires_grad=True)
        M = Tensor(np.zeros((3, 4)), requires_grad=True)
        (r + M).sum().backward()
        self.assertEqual(r.grad.shape, (1, 4))
        self.assertTrue(allclose(r.grad, np.full((1, 4), 3.0)))

if __name__ == "__main__":
    unittest.main(verbosity=2)