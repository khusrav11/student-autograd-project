import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import unittest
from src.tensor import Tensor

def allclose(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)

class TestMatMul(unittest.TestCase):

    def test_square_shapes(self):
        A = Tensor(np.eye(2), requires_grad=True)
        B = Tensor(np.ones((2, 2)), requires_grad=True)
        C = A @ B
        self.assertEqual(C.shape, (2, 2))

    def test_square_forward(self):
        A = Tensor(np.eye(2), requires_grad=True)
        B = Tensor(np.ones((2, 2)), requires_grad=True)
        C = A @ B
        self.assertTrue(allclose(C.data, np.ones((2, 2))))

    def test_square_backward_A(self):
        A = Tensor(np.array([[1.0, 2.0],[3.0, 4.0]]), requires_grad=True)
        B = Tensor(np.array([[5.0, 6.0],[7.0, 8.0]]), requires_grad=True)
        (A @ B).sum().backward()
        expected = np.ones((2, 2)) @ B.data.T
        self.assertTrue(allclose(A.grad, expected))

    def test_square_backward_B(self):
        A = Tensor(np.array([[1.0, 2.0],[3.0, 4.0]]), requires_grad=True)
        B = Tensor(np.array([[5.0, 6.0],[7.0, 8.0]]), requires_grad=True)
        (A @ B).sum().backward()
        expected = A.data.T @ np.ones((2, 2))
        self.assertTrue(allclose(B.grad, expected))

    def test_nonsquare_output_shape(self):
        """(5,4) @ (4,3) → (5,3)"""
        X = Tensor(np.random.randn(5, 4), requires_grad=True)
        W = Tensor(np.random.randn(4, 3), requires_grad=True)
        self.assertEqual((X @ W).shape, (5, 3))

    def test_nonsquare_grad_shapes(self):
        X = Tensor(np.random.randn(5, 4), requires_grad=True)
        W = Tensor(np.random.randn(4, 3), requires_grad=True)
        (X @ W).sum().backward()
        self.assertEqual(X.grad.shape, (5, 4))
        self.assertEqual(W.grad.shape, (4, 3))

    def test_nonsquare_grad_X_values(self):
        np.random.seed(99)
        X = Tensor(np.random.randn(5, 4), requires_grad=True)
        W = Tensor(np.random.randn(4, 3), requires_grad=True)
        (X @ W).sum().backward()
        expected = np.ones((5, 3)) @ W.data.T
        self.assertTrue(allclose(X.grad, expected))

    def test_nonsquare_grad_W_values(self):
        np.random.seed(99)
        X = Tensor(np.random.randn(5, 4), requires_grad=True)
        W = Tensor(np.random.randn(4, 3), requires_grad=True)
        (X @ W).sum().backward()
        expected = X.data.T @ np.ones((5, 3))
        self.assertTrue(allclose(W.grad, expected))

    def test_linear_layer_bias_shapes(self):
        # X @ W + b: all grad shapes should match parameter shapes
        X = Tensor(np.random.randn(8, 4), requires_grad=True)
        W = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.zeros(3), requires_grad=True)
        (X @ W + b).sum().backward()
        self.assertEqual(X.grad.shape, (8, 4))
        self.assertEqual(W.grad.shape, (4, 3))
        self.assertEqual(b.grad.shape, (3,))

    def test_numerical_gradient_X(self):
        # quick finite-diff check for X gradient
        np.random.seed(7)
        X_data = np.random.randn(3, 4)
        W_data = np.random.randn(4, 2)
        eps = 1e-5
        X = Tensor(X_data.copy(), requires_grad=True)
        W = Tensor(W_data.copy(), requires_grad=True)
        (X @ W).sum().backward()
        analytical = X.grad.copy()
        numerical = np.zeros_like(X_data)
        for i in range(X_data.shape[0]):
            for j in range(X_data.shape[1]):
                Xp, Xm = X_data.copy(), X_data.copy()
                Xp[i, j] += eps; Xm[i, j] -= eps
                numerical[i, j] = (np.sum(Xp @ W_data) - np.sum(Xm @ W_data)) / (2 * eps)
        self.assertTrue(allclose(analytical, numerical, atol=1e-4))

    def test_no_grad(self):
        A = Tensor(np.ones((2, 3)), requires_grad=False)
        B = Tensor(np.ones((3, 2)), requires_grad=False)
        self.assertFalse((A @ B).requires_grad)

if __name__ == "__main__":
    unittest.main(verbosity=2)