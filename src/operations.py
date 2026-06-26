import numpy as np
from src.tensor import Tensor, _ensure_tensor

# Elementwise operations
def exp(x):
    return _ensure_tensor(x).exp()

def log(x):
    return _ensure_tensor(x).log()

def tanh(x):
    return _ensure_tensor(x).tanh()

def relu(x):
    return _ensure_tensor(x).relu()

def sigmoid(x):
    x = _ensure_tensor(x)
    return Tensor(1.0) / (Tensor(1.0) + (-x).exp())

# Reduction operations
def sum(x, axis=None, keepdims=False):
    return _ensure_tensor(x).sum(axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return _ensure_tensor(x).mean(axis=axis, keepdims=keepdims)

# Matrix operations
def matmul(a, b):
    return _ensure_tensor(a) @ _ensure_tensor(b)

# Loss functions
def softmax(logits):
    logits = _ensure_tensor(logits)
    # computing in NumPy directly for numerical stability but still need the backward so I build it from Tensor ops
    shifted = logits - Tensor(np.max(logits.data, axis=-1, keepdims=True))
    exp_vals = shifted.exp()
    return exp_vals / exp_vals.sum(axis=-1, keepdims=True)

def mse_loss(predictions, targets):
    predictions = _ensure_tensor(predictions)
    targets = _ensure_tensor(targets)
    diff = predictions - targets
    return (diff ** 2).mean()

# Parameter initialisation helpers
def he_init(shape, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, shape)

def xavier_init(shape, rng=None):
    # Xavier (Glorot) initialisation, variance = 1 / fan_in
    if rng is None:
        rng = np.random.default_rng()
    fan_in = shape[0]
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, shape)