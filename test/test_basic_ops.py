import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import unittest
from src.tensor import Tensor

def allclose(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)

class TestAddition(unittest.TestCase):
    def test_scalar_add(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(4.0, requires_grad=True)
        z = x + y
        z.backward()
        self.assertTrue(allclose(z.data, 7.0))
        self.assertTrue(allclose(x.grad, 1.0))
        self.assertTrue(allclose(y.grad, 1.0))

    def test_vector_add(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a + b).sum()
        c.backward()
        self.assertTrue(allclose(c.data, 21.0))
        self.assertTrue(allclose(a.grad, np.ones(3)))
        self.assertTrue(allclose(b.grad, np.ones(3)))

    def test_radd(self):
        x = Tensor(5.0, requires_grad=True)
        z = 3.0 + x
        z.backward()
        self.assertTrue(allclose(z.data, 8.0))
        self.assertTrue(allclose(x.grad, 1.0))

class TestSubtraction(unittest.TestCase):
    def test_scalar_sub(self):
        x = Tensor(10.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = x - y
        z.backward()
        self.assertTrue(allclose(z.data, 7.0))
        self.assertTrue(allclose(x.grad, 1.0))
        self.assertTrue(allclose(y.grad, -1.0))

    def test_neg(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = (-x).sum()
        z.backward()
        self.assertTrue(allclose(x.grad, -np.ones(3)))

class TestMultiplication(unittest.TestCase):
    def test_scalar_mul(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(4.0, requires_grad=True)
        z = x * y
        z.backward()
        self.assertTrue(allclose(z.data, 12.0))
        self.assertTrue(allclose(x.grad, 4.0))
        self.assertTrue(allclose(y.grad, 3.0))

    def test_scalar_constant(self):
        x = Tensor(5.0, requires_grad=True)
        z = x * 3.0
        z.backward()
        self.assertTrue(allclose(x.grad, 3.0))

    def test_elementwise_vector(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a * b).sum()
        c.backward()
        self.assertTrue(allclose(a.grad, np.array([4.0, 5.0, 6.0])))
        self.assertTrue(allclose(b.grad, np.array([1.0, 2.0, 3.0])))

class TestDivision(unittest.TestCase):
    def test_scalar_div(self):
        x = Tensor(6.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x / y
        z.backward()
        self.assertTrue(allclose(z.data, 3.0))
        self.assertTrue(allclose(x.grad, 0.5))
        self.assertTrue(allclose(y.grad, -1.5))

class TestPower(unittest.TestCase):
    def test_square(self):
        x = Tensor(3.0, requires_grad=True)
        z = x ** 2
        z.backward()
        self.assertTrue(allclose(z.data, 9.0))
        self.assertTrue(allclose(x.grad, 6.0))

    def test_cube_vector(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = (x ** 3).sum()
        z.backward()
        self.assertTrue(allclose(x.grad, 3.0 * x.data ** 2))

class TestExp(unittest.TestCase):
    def test_exp_scalar(self):
        x = Tensor(1.0, requires_grad=True)
        z = x.exp()
        z.backward()
        self.assertTrue(allclose(z.data, np.e))
        self.assertTrue(allclose(x.grad, np.e))

    def test_exp_vector(self):
        x = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)
        z = x.exp().sum()
        z.backward()
        self.assertTrue(allclose(x.grad, np.exp([0.0, 1.0, 2.0])))

class TestLog(unittest.TestCase):
    def test_log_scalar(self):
        x = Tensor(np.e, requires_grad=True)
        z = x.log()
        z.backward()
        self.assertTrue(allclose(z.data, 1.0))
        self.assertTrue(allclose(x.grad, 1.0 / np.e))

    def test_log_stability(self):
        x = Tensor(1e-15, requires_grad=True)
        z = x.log()
        self.assertTrue(np.isfinite(z.data))

class TestTanh(unittest.TestCase):
    def test_tanh_scalar(self):
        x = Tensor(0.5, requires_grad=True)
        z = x.tanh()
        z.backward()
        t = np.tanh(0.5)
        self.assertTrue(allclose(z.data, t))
        self.assertTrue(allclose(x.grad, 1 - t ** 2))

    def test_tanh_zero(self):
        x = Tensor(0.0, requires_grad=True)
        z = x.tanh()
        z.backward()
        self.assertTrue(allclose(z.data, 0.0))
        self.assertTrue(allclose(x.grad, 1.0))

class TestReLU(unittest.TestCase):
    def test_relu_positive(self):
        x = Tensor(3.0, requires_grad=True)
        z = x.relu()
        z.backward()
        self.assertTrue(allclose(z.data, 3.0))
        self.assertTrue(allclose(x.grad, 1.0))

    def test_relu_negative(self):
        x = Tensor(-2.0, requires_grad=True)
        z = x.relu()
        z.backward()
        self.assertTrue(allclose(z.data, 0.0))
        self.assertTrue(allclose(x.grad, 0.0))

    def test_relu_vector(self):
        x = Tensor(np.array([-3.0, 0.0, 2.0, -1.0, 4.0]), requires_grad=True)
        z = x.relu().sum()
        z.backward()
        self.assertTrue(allclose(x.grad, np.array([0.0, 0.0, 1.0, 0.0, 1.0])))

class TestReductions(unittest.TestCase):
    def test_sum_all(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = x.sum()
        z.backward()
        self.assertTrue(allclose(z.data, 6.0))
        self.assertTrue(allclose(x.grad, np.ones(3)))

    def test_mean_all(self):
        x = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)
        z = x.mean()
        z.backward()
        self.assertTrue(allclose(z.data, 4.0))
        self.assertTrue(allclose(x.grad, np.full(3, 1.0 / 3)))

class TestComposedOps(unittest.TestCase):
    def test_chain_rule(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = (x + y) * x
        z.backward()
        self.assertTrue(allclose(x.grad, 8.0))
        self.assertTrue(allclose(y.grad, 3.0))

    def test_reuse_same_tensor(self):
        """Gradient accumulation: x used twice in x*x, grad should be 2x not x."""
        x = Tensor(4.0, requires_grad=True)
        z = x * x
        z.backward()
        self.assertTrue(allclose(x.grad, 8.0))

    def test_deep_chain(self):
        x = Tensor(np.array([0.1, -0.5, 0.3]), requires_grad=True)
        z = x.exp().tanh().relu().mean()
        z.backward()
        self.assertTrue(np.all(np.isfinite(x.grad)))

if __name__ == "__main__":
    unittest.main(verbosity=2)
