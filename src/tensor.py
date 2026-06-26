import numpy as np

class Tensor:
    # сomputation graph node: wraps a numpy array and tracks gradients
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        # always store data as a float64 numpy array for consistency
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        # grad starts as None it will be created lazily as a zeros array
        # with the same shape as self.data once backward is triggered
        self.grad = None
        # internal fields for the computation graph
        self._parents = set(_parents)  # set of parent Tensors
        self._backward = lambda: None   # filled in by each operation
        self._op = _op                  # just for debugging / __repr__
    
    # shape helpers
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, op='{self._op}', "f"requires_grad={self.requires_grad})\n", f"data={self.data}")
    
    # gradient utilities
    def zero_grad(self):
        # reset accumulated gradient to zero (call before each optimizer step)
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def _init_grad(self):
        # lazily initialise grad to zeros if it hasn't been set yet
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    # backward pass
    def backward(self):
        # reverse-mode autograd: builds computation graph, sorts it topologically, then walks it in reverse calling each node's _backward to accumulate gradients
        # step 1 and 2: topological sort
        topo = []
        visited = set()
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._parents:
                    build_topo(parent)
                topo.append(node)
        build_topo(self)
        # step 3: seed the gradient at the root
        self._init_grad()
        self.grad = np.ones_like(self.data, dtype=np.float64)
        # step 4: propagate backwards
        for node in reversed(topo):
            node._backward()

    # basic arithmetic operators
    def __add__(self, other):
        # elementwise addition, with broadcasting support
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="add",)

        def _backward():
            # gradient of addition is 1 for both inputs.
            # if broadcasting happened, we need to sum along broadcast axes to reduce the gradient back to the original shape.
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # elementwise subtraction
        other = _ensure_tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="sub",)

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other._init_grad()
                # subtraction: gradient for the right operand is negated
                other.grad += _unbroadcast(-out.grad, other.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        # elementwise multiplication 
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="mul",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other._init_grad()
                other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # elementwise division
        other = _ensure_tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="div",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                # d/dx [x/y] = 1/y
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other._init_grad()
                # d/dy [x/y] = -x / y^2
                other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape,)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return _ensure_tensor(other).__truediv__(self)

    def __neg__(self):
        return self * -1.0

    def __pow__(self, exponent):
        # raise tensor to a scalar power, d/dx [x^n] = n * x^(n-1)
        assert isinstance(exponent, (int, float)), "Exponent must be a scalar"
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad, _parents=(self,), _op=f"pow({exponent})",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * exponent * (self.data ** (exponent - 1))
        out._backward = _backward
        return out

    # matrix multiplication
    def __matmul__(self, other):
        # matrix multiplication: (m,k) @ (k,n) → (m,n), dL/dself = dL/dout @ other.T,  dL/dother = self.T @ dL/dout
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _parents=(self, other), _op="matmul",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._init_grad()
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # reduction operations
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _parents=(self,), _op="sum",)
        original_shape = self.shape
        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad
                # if we reduced without keepdims, we need to re-expand axes
                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                # now broadcast to original shape
                self.grad += np.broadcast_to(grad, original_shape).copy()
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _parents=(self,), _op="mean",)
        original_shape = self.shape
        # computing N = number of elements involved in the mean
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = 1
            for ax in axes:
                n *= self.shape[ax]
        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = out.grad / n  # scale by 1/N
                if not keepdims and axis is not None:
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, original_shape).copy()
        out._backward = _backward
        return out

    # elementwise nonlinearities
    def exp(self):
        exp_val = np.exp(self.data)
        out = Tensor(exp_val, requires_grad=self.requires_grad, _parents=(self,), _op="exp",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * exp_val  # exp_val = e^x stored above
        out._backward = _backward
        return out

    def log(self):
        clipped = np.clip(self.data, 1e-12, None)  # numerical safety
        out = Tensor(np.log(clipped), requires_grad=self.requires_grad, _parents=(self,), _op="log",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad / clipped
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _parents=(self,), _op="tanh",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad * (1.0 - t ** 2)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), requires_grad=self.requires_grad, _parents=(self,), _op="relu",)
        def _backward():
            if self.requires_grad:
                self._init_grad()
                # Gradient mask: 1 where input was positive
                self.grad += out.grad * (self.data > 0).astype(np.float64)
        out._backward = _backward
        return out

    # convenience: allow indexing (read-only, no grad through slice)
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _parents=(self,), _op="index",)
        original_shape = self.shape
        def _backward():
            if self.requires_grad:
                self._init_grad()
                grad = np.zeros(original_shape, dtype=np.float64)
                np.add.at(grad, idx, out.grad)
                self.grad += grad
        out._backward = _backward
        return out

# module-level helpers
def _ensure_tensor(x):
    # wraping the scalars and numpy arrays in a Tensor if they are not already
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def _unbroadcast(grad, target_shape):
    # after a broadcasted op, gradient has the output shape.
    # reduces it back to target_shape by summing over axes that were broadcast.
    # example: output (64, 3) + bias (3,) → sum axis 0 to get gradient shape (3,).
    if grad.shape == target_shape:
        return grad  # fast path: no broadcasting happened
    # make grad the same number of dimensions as target_shape by summing, over leading axes that don't exist in target_shape
    ndim_out = grad.ndim
    ndim_tgt = len(target_shape)
    # sum over extra leading dimensions
    if ndim_out > ndim_tgt:
        grad = grad.sum(axis=tuple(range(ndim_out - ndim_tgt)))
    # now both have the same number of dimensions, sum over axes where target was size 1 (those were broadcast).
    assert grad.ndim == len(target_shape), "Shape mismatch after leading sum"
    axes_to_sum = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape))
        if t_dim == 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

    return grad
