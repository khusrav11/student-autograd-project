import numpy as np
from src.tensor import Tensor, _ensure_tensor

def softmax_cross_entropy_loss(logits, labels):
    # numerically stable softmax cross-entropy loss for multi-class classification
    logits = _ensure_tensor(logits) # Tensor of shape (batch_size, n_classes)
    labels = np.array(labels, dtype=np.int64) # Array like of shape (batch_size,)
    batch_size, n_classes = logits.data.shape
    # forward: numerically stable cross-entropy
    # subtracting row max for numerical stability
    z = logits.data
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(z_shifted), axis=1)) 
    # picking the logit for the correct class
    correct_logits = z_shifted[np.arange(batch_size), labels] 
    per_sample_loss = -correct_logits + log_sum_exp           
    loss_val = np.mean(per_sample_loss)

    out = Tensor(loss_val, requires_grad=logits.requires_grad, _parents=(logits,), _op="softmax_cross_ent",)

    def _backward():
        if logits.requires_grad:
            logits._init_grad()
            # calculating the softmax
            probs = np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)
            grad = probs.copy()
            grad[np.arange(batch_size), labels] -= 1.0
            grad /= batch_size          # because we took the mean
            logits.grad += out.grad * grad   # chain rule: multiply by upstream grad
    out._backward = _backward
    return out

def mse_loss(predictions, targets):
    # L = mean((predictions - targets)^2)
    predictions = _ensure_tensor(predictions)
    targets = _ensure_tensor(targets)
    diff = predictions - targets
    return (diff ** 2).mean()

def binary_cross_entropy(probs, labels):
    # for two-class problems (sigmoid output)
    # L = -mean(y * log(p) + (1-y) * log(1-p))
    probs = _ensure_tensor(probs)
    labels = _ensure_tensor(labels)
    return -(labels * probs.log() + (Tensor(1.0) - labels) * (Tensor(1.0) - probs).log()).mean()