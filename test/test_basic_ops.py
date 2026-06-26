import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import unittest
from src.tensor import Tensor

# tolerance for float comparisons
def allclose(a, b, atol=1e-6):
    return np.allclose(a, b, atol=atol)

class TestAddition(unittest.TestCase):
    def test_scalar_add(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(4.0, requires_grad=True)
        z = x + y
        z.backward()
        self.assertTrue(allclose(z.data, 7.0))        # forward: 3+4=7
        self.assertTrue(allclose(x.grad, 1.0))        # dz/dx = 1
        self.assertTrue(allclose(y.grad, 1.0))        # dz/dy = 1

    def test_vector_add(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a + b).sum()                             # reduce to scalar before backward
        c.backward()
        self.assertTrue(allclose(c.data, 21.0))
        self.assertTrue(allclose(a.grad, np.ones(3))) # grad of sum wrt each element = 1

    def test_radd(self):
        x = Tensor(5.0, requires_grad=True)
        z = 3.0 + x                                   # tests __radd__: constant on the left
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
        self.assertTrue(allclose(x.grad,  1.0))       # dz/dx = +1
        self.assertTrue(allclose(y.grad, -1.0))       # dz/dy = -1

    def test_neg(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = (-x).sum()                                # tests __neg__
        z.backward()
        self.assertTrue(allclose(x.grad, -np.ones(3)))

class TestMultiplication(unittest.TestCase):
    def test_scalar_mul(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(4.0, requires_grad=True)
        z = x * y
        z.backward()
        self.assertTrue(allclose(z.data, 12.0))
        self.assertTrue(allclose(x.grad, 4.0))        # dz/dx = y = 4
        self.assertTrue(allclose(y.grad, 3.0))        # dz/dy = x = 3

    def test_scalar_constant(self):
        x = Tensor(5.0, requires_grad=True)
        z = x * 3.0                                   # multiply by plain Python float
        z.backward()
        self.assertTrue(allclose(x.grad, 3.0))

    def test_elementwise_vector(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
        c = (a * b).sum()
        c.backward()
        self.assertTrue(allclose(a.grad, np.array([4.0, 5.0, 6.0])))  # grad wrt a = b
        self.assertTrue(allclose(b.grad, np.array([1.0, 2.0, 3.0])))  # grad wrt b = a

class TestDivision(unittest.TestCase):
    def test_scalar_div(self):
        x = Tensor(6.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = x / y
        z.backward()
        self.assertTrue(allclose(z.data, 3.0))
        self.assertTrue(allclose(x.grad,  0.5))       # dz/dx = 1/y = 0.5
        self.assertTrue(allclose(y.grad, -1.5))       # dz/dy = -x/y² = -1.5

class TestPower(unittest.TestCase):
    def test_square(self):
        x = Tensor(3.0, requires_grad=True)
        z = x ** 2
        z.backward()
        self.assertTrue(allclose(z.data, 9.0))
        self.assertTrue(allclose(x.grad, 6.0))        # dz/dx = 2x = 6

    def test_cube_vector(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = (x ** 3).sum()
        z.backward()
        self.assertTrue(allclose(x.grad, 3.0 * x.data ** 2))  # dz/dx = 3x²

class TestExp(unittest.TestCase):
    def test_exp_scalar(self):
        x = Tensor(1.0, requires_grad=True)
        z = x.exp()
        z.backward()
        self.assertTrue(allclose(z.data, np.e))       # e^1 = e
        self.assertTrue(allclose(x.grad, np.e))       # d(e^x)/dx = e^x = e

    def test_exp_vector(self):
        x = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)
        z = x.exp().sum()
        z.backward()
        self.assertTrue(allclose(x.grad, np.exp([0.0, 1.0, 2.0])))  # grad = e^x elementwise

class TestLog(unittest.TestCase):
    def test_log_scalar(self):
        x = Tensor(np.e, requires_grad=True)
        z = x.log()
        z.backward()
        self.assertTrue(allclose(z.data, 1.0))        # log(e) = 1
        self.assertTrue(allclose(x.grad, 1.0 / np.e)) # d(log x)/dx = 1/x

    def test_log_stability(self):
        x = Tensor(1e-15, requires_grad=True)
        z = x.log()
        self.assertTrue(np.isfinite(z.data))          # must not return -inf or nan

class TestTanh(unittest.TestCase):
    def test_tanh_scalar(self):
        x = Tensor(0.5, requires_grad=True)
        z = x.tanh()
        z.backward()
        t = np.tanh(0.5)
        self.assertTrue(allclose(z.data, t))
        self.assertTrue(allclose(x.grad, 1 - t ** 2)) # d(tanh)/dx = 1 - tanh²(x)

    def test_tanh_zero(self):
        x = Tensor(0.0, requires_grad=True)
        z = x.tanh()
        z.backward()
        self.assertTrue(allclose(z.data, 0.0))        # tanh(0) = 0
        self.assertTrue(allclose(x.grad, 1.0))        # grad at 0 = 1 (maximum slope)

class TestReLU(unittest.TestCase):
    def test_relu_positive(self):
        x = Tensor(3.0, requires_grad=True)
        z = x.relu()
        z.backward()
        self.assertTrue(allclose(z.data, 3.0))        # relu(3) = 3
        self.assertTrue(allclose(x.grad, 1.0))        # grad = 1 for x > 0

    def test_relu_negative(self):
        x = Tensor(-2.0, requires_grad=True)
        z = x.relu()
        z.backward()
        self.assertTrue(allclose(z.data, 0.0))        # relu(-2) = 0
        self.assertTrue(allclose(x.grad, 0.0))        # grad = 0 for x < 0 (dead neuron)

    def test_relu_vector(self):
        x = Tensor(np.array([-3.0, 0.0, 2.0, -1.0, 4.0]), requires_grad=True)
        z = x.relu().sum()
        z.backward()
        # only positive elements pass gradient through
        self.assertTrue(allclose(x.grad, np.array([0.0, 0.0, 1.0, 0.0, 1.0])))

class TestReductions(unittest.TestCase):
    def test_sum_all(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        z = x.sum()
        z.backward()
        self.assertTrue(allclose(z.data, 6.0))
        self.assertTrue(allclose(x.grad, np.ones(3)))        # dsum/dx_i = 1 for all i

    def test_mean_all(self):
        x = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)
        z = x.mean()
        z.backward()
        self.assertTrue(allclose(z.data, 4.0))
        self.assertTrue(allclose(x.grad, np.full(3, 1.0 / 3)))  # dmean/dx_i = 1/N

class TestComposedOps(unittest.TestCase):
    def test_chain_rule(self):
        x = Tensor(3.0, requires_grad=True)
        y = Tensor(2.0, requires_grad=True)
        z = (x + y) * x                              # z = x² + xy
        z.backward()
        self.assertTrue(allclose(x.grad, 8.0))       # dz/dx = 2x + y = 8
        self.assertTrue(allclose(y.grad, 3.0))       # dz/dy = x = 3

    def test_reuse_same_tensor(self):
        """Gradient accumulation: x used twice in x*x, grad should be 2x not x."""
        x = Tensor(4.0, requires_grad=True)
        z = x * x                                    # same tensor on both sides
        z.backward()
        self.assertTrue(allclose(x.grad, 8.0))       # grads from both paths accumulate: x+x = 2x

    def test_deep_chain(self):
        x = Tensor(np.array([0.1, -0.5, 0.3]), requires_grad=True)
        z = x.exp().tanh().relu().mean()             # 4-op chain: exp→tanh→relu→mean
        z.backward()
        self.assertTrue(np.all(np.isfinite(x.grad))) # no nan/inf through the whole chain

if __name__ == "__main__":
    unittest.main(verbosity=2)
