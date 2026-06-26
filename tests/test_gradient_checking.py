import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import unittest
from src.tensor import Tensor
from src.losses import softmax_cross_entropy_loss

def grad_check(function, inputs, eps=1e-5, tolerance=1e-4):
    # central finite-difference: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε
    for inp in inputs:
        inp.zero_grad()
    output = function(*inputs)
    output.backward()
    # save analytical grads before any data is perturbed
    analytical_grads = [inp.grad.copy() for inp in inputs]

    numerical_grads = []
    for inp in inputs:
        num_grad = np.zeros_like(inp.data, dtype=np.float64)
        it = np.nditer(inp.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = inp.data[idx]
            # perturb +ε
            inp.data[idx] = original + eps
            for i in inputs: i.zero_grad()
            fp = function(*inputs).data.item()
            # perturb -ε
            inp.data[idx] = original - eps
            for i in inputs: i.zero_grad()
            fm = function(*inputs).data.item()
            # restore original value
            inp.data[idx] = original
            num_grad[idx] = (fp - fm) / (2.0 * eps)
            it.iternext()
        numerical_grads.append(num_grad)

    max_rel_err = 0.0
    max_abs_err = 0.0
    for analytic, numeric in zip(analytical_grads, numerical_grads):
        abs_err = np.abs(analytic - numeric)
        # relative error: |a-n| / (|a|+|n|), clamped to avoid div-by-zero
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
        rel_err = abs_err / denom
        max_rel_err = max(max_rel_err, np.max(rel_err))
        max_abs_err = max(max_abs_err, np.max(abs_err))

    passed = max_rel_err < tolerance
    return max_abs_err, max_rel_err, passed


class TestRequiredGradChecks(unittest.TestCase):
    # helper: runs grad_check and fails the test if rel_err >= tolerance
    def _check(self, name, fn, inputs):
        abs_e, rel_e, passed = grad_check(fn, inputs)
        print(f"\nTest: {name}")
        print(f"  Max abs error: {abs_e:.2e}  Max rel error: {rel_e:.2e}  {'PASS' if passed else 'FAIL'}")
        self.assertTrue(passed, f"Gradient check FAILED for '{name}' (rel_err={rel_e:.2e})")

    def test_01_scalar_expression(self):
        # simplest case: scalar x*y+z, grads = [y, x, 1]
        x = Tensor(np.array(3.0), requires_grad=True)
        y = Tensor(np.array(4.0), requires_grad=True)
        z = Tensor(np.array(2.0), requires_grad=True)
        self._check("scalar x*y+z", lambda x,y,z: x*y+z, [x,y,z])

    def test_02_vector_elementwise(self):
        # tests ** and exp in the same graph, then sum to scalar
        x = Tensor(np.array([1.0,2.0,3.0]), requires_grad=True)
        y = Tensor(np.array([0.1,-0.5,0.8]), requires_grad=True)
        self._check("vector x^2+exp(y)", lambda x,y: (x**2+y.exp()).sum(), [x,y])

    def test_03_matmul_with_bias(self):
        # checks matmul grad AND bias broadcast grad in one shot
        np.random.seed(42)
        X = Tensor(np.random.randn(5,4), requires_grad=True)
        W = Tensor(np.random.randn(4,3), requires_grad=True)
        b = Tensor(np.random.randn(3), requires_grad=True)
        self._check("matmul X@W+b", lambda X,W,b: (X@W+b).sum(), [X,W,b])

    def test_04_mse_loss(self):
        # target is fixed numpy array; only pred is differentiated
        # expected grad: 2*(pred-tgt)/N
        np.random.seed(1)
        pred = Tensor(np.random.randn(10), requires_grad=True)
        tgt_data = np.random.randn(10)
        def fn(pred):
            tgt = Tensor(tgt_data)
            return ((pred - tgt) ** 2).mean()
        self._check("MSE loss", fn, [pred])

    def test_05_softmax_cross_entropy(self):
        # most complex required check: softmax + log + gather + mean
        np.random.seed(7)
        logits = Tensor(np.random.randn(8,3), requires_grad=True)
        labels = np.array([0,1,2,0,1,2,0,1])
        self._check("softmax CE", lambda l: softmax_cross_entropy_loss(l, labels), [logits])


class TestCustomGradChecks(unittest.TestCase):
    def _check(self, name, fn, inputs):
        abs_e, rel_e, passed = grad_check(fn, inputs)
        print(f"\nCustom: {name}")
        print(f"  Max abs error: {abs_e:.2e}  Max rel error: {rel_e:.2e}  {'PASS' if passed else 'FAIL'}")
        self.assertTrue(passed, f"Custom grad check FAILED for '{name}' (rel_err={rel_e:.2e})")

    def test_c01_tanh_chain(self):
        # chain: mul → tanh → mean; tests tanh backward through product
        np.random.seed(10)
        x = Tensor(np.random.randn(4,3)*0.5, requires_grad=True)
        y = Tensor(np.random.randn(4,3)*0.5, requires_grad=True)
        self._check("tanh(x*y).mean", lambda x,y: (x*y).tanh().mean(), [x,y])

    def test_c02_relu_matmul(self):
        # relu grad is a binary mask; matmul grad = upstream @ W.T and X.T @ upstream
        np.random.seed(20)
        X = Tensor(np.random.randn(6,3), requires_grad=True)
        W = Tensor(np.random.randn(3,4), requires_grad=True)
        self._check("relu(X@W).sum", lambda X,W: (X@W).relu().sum(), [X,W])

    def test_c03_log_of_softmax_probs(self):
        # manual softmax (shift → exp → normalize → log): tests div and log backward
        np.random.seed(30)
        x = Tensor(np.random.randn(5,4), requires_grad=True)
        def fn(x):
            shifted = x - Tensor(np.max(x.data, axis=1, keepdims=True))  # numerical stability
            e = shifted.exp()
            probs = e / e.sum(axis=1, keepdims=True)
            return probs.log().sum()
        self._check("log(softmax(x)).sum", fn, [x])

    def test_c04_two_layer_mlp(self):
        # full forward pass of a 2-layer MLP: tests all ops together end-to-end
        np.random.seed(50)
        X  = Tensor(np.random.randn(4,2), requires_grad=True)
        W1 = Tensor(np.random.randn(2,4)*0.5, requires_grad=True)
        b1 = Tensor(np.zeros(4), requires_grad=True)
        W2 = Tensor(np.random.randn(4,3)*0.5, requires_grad=True)
        b2 = Tensor(np.zeros(3), requires_grad=True)
        def fn(X,W1,b1,W2,b2):
            return ((X@W1+b1).relu()@W2+b2).sum()
        self._check("2-layer MLP forward", fn, [X,W1,b1,W2,b2])

    def test_c05_broadcast_mul(self):
        # (5,1)*(1,4) → (5,4): grads must be summed back to original shapes
        np.random.seed(60)
        a = Tensor(np.random.randn(5,1), requires_grad=True)
        b = Tensor(np.random.randn(1,4), requires_grad=True)
        self._check("broadcast (5,1)*(1,4)", lambda a,b: (a*b).sum(), [a,b])

    def test_c06_div_then_exp(self):
        # tests div backward (quotient rule) composed with exp backward
        np.random.seed(70)
        x = Tensor(np.random.randn(3,3)*0.5, requires_grad=True)
        y = Tensor(np.abs(np.random.randn(3,3))+0.5, requires_grad=True)  # y > 0 to avoid div-by-zero
        self._check("exp(x/y).sum", lambda x,y: (x/y).exp().sum(), [x,y])

    def test_c07_large_logits_stability(self):
        # logits scaled ×50: true grads ≈ 0, so rel error is meaningless
        # use abs error < 1e-8 as the pass criterion instead
        np.random.seed(80)
        logits = Tensor(np.random.randn(5,3)*50.0, requires_grad=True)
        labels = np.array([0,1,2,0,1])
        abs_e, rel_e, _ = grad_check(
            lambda l: softmax_cross_entropy_loss(l, labels), [logits],
            tolerance=0.1  # relaxed tolerance; abs check below is the real gate
        )
        print(f"\nCustom: softmax CE large logits")
        print(f"  Max abs error: {abs_e:.2e}  Max rel error: {rel_e:.2e}")
        self.assertLess(abs_e, 1e-8, f"Absolute error too large: {abs_e:.2e}")
        print("  PASS (abs error < 1e-8: gradients are numerically zero and correct)")

    def test_c08_mean_axis_then_mul(self):
        # mean(x, axis=0) collapses rows; result (4,) is multiplied elementwise by y
        np.random.seed(90)
        x = Tensor(np.random.randn(6,4), requires_grad=True)
        y = Tensor(np.random.randn(4), requires_grad=True)
        self._check("(mean(x,0)*y).sum", lambda x,y: (x.mean(axis=0)*y).sum(), [x,y])

    def test_c09_power_chain(self):
        # (relu(x)+1)^3: tests power backward through relu and scalar add
        np.random.seed(100)
        x = Tensor(np.random.randn(5)*0.5, requires_grad=True)
        self._check("((relu(x)+1)^3).sum", lambda x: ((x.relu()+Tensor(1.0))**3).sum(), [x])

    def test_c10_nested_relu_tanh(self):
        # relu then tanh: dead neurons (x<0) must have zero grad through both
        np.random.seed(111)
        x = Tensor(np.random.randn(4,3), requires_grad=True)
        self._check("tanh(relu(x)).mean", lambda x: x.relu().tanh().mean(), [x])


def run_report(save_path=None):
    # runs all tests programmatically and writes a formatted pass/fail report
    print("\n" + "="*60)
    print("GRADIENT CHECK REPORT")
    print("="*60)
    import io
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRequiredGradChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomGradChecks))
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)
    result = runner.run(suite)
    report_text = buf.getvalue()
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    summary = (
        f"\nSUMMARY: {passed}/{total} gradient checks passed.\n"
        f"{'All checks PASSED' if passed==total else 'Some checks FAILED'}\n"
    )
    print(report_text)
    print(summary)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text + summary)
        print(f"Report saved to {save_path}")
    return passed == total


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # runs all 15 grad checks and saves results/gradient_check_results.txt
    run_report(save_path="results/gradient_check_results.txt")