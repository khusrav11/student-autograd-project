# Final Report: Mini Automatic Differentiation Engine
**Course:** Machine Learning
**Project:** Mini Autograd Engine and Neural Network Training
**Author:** Ziyoboev Khusravkhon
**Date:** 26.06.2026

---

## 1. Project Overview

This project implements a complete automatic differentiation (autograd) engine from scratch using only NumPy. The engine supports scalar, vector, and matrix operations, correctly handles broadcasting, and passes finite-difference gradient checks for all required operations. It is then used to train a two-layer MLP on a 3-class spiral classification dataset.

**Key results:**
- All 15 gradient checks pass (5 required + 10 custom)
- Final training accuracy: **≥ 90%** on the spiral dataset
- Debugging investigation completed (missing `zero_grad` bug identified and fixed)

---

## 2. Automatic Differentiation Design

### 2.1 What is Autograd?

Automatic differentiation computes exact derivatives of functions defined by code. Unlike symbolic differentiation (manipulates formulas, expensive) or numerical differentiation (finite differences, imprecise), autograd records a **computation graph** during the forward pass — a DAG where nodes are tensors and edges are operations. The backward pass traverses it in reverse, applying the chain rule at each node.

### 2.2 The `Tensor` Class

Each `Tensor` stores:
- `data` — float64 NumPy array
- `grad` — accumulated gradient dLoss/dSelf
- `_parents` — inputs that created this tensor
- `_backward` — closure that propagates gradient to parents
- `requires_grad` — whether to track gradients

### 2.3 Forward Pass & Graph Construction

Every operation creates a new `Tensor` and registers a `_backward` closure. For example:

```python
z = x * y   # new Tensor with _parents = {x, y}
            # _backward computes: x.grad += out.grad * y.data
```

### 2.4 Backward Pass (Reverse-Mode AD)

`loss.backward()` does three things:
1. **Topological sort** — DFS from root, reversed so every node comes after its downstream dependencies
2. **Seed gradient** — `loss.grad = 1.0` (dLoss/dLoss = 1)
3. **Propagate** — walk nodes in reverse topological order, calling each `_backward`

**Why topological order?** A node's gradient depends on all downstream nodes being computed first. Processing out of order gives incomplete gradients.

**Why accumulate (`+=`) instead of overwrite (`=`)?** If tensor `x` feeds into multiple operations, both paths contribute to its gradient. The total is the sum over all paths (multivariate chain rule). Overwriting silently discards contributions.

---

## 3. Broadcasting Backward Pass

When a bias of shape `(3,)` is added to a batch of shape `(64, 3)`, NumPy broadcasts it to `(64, 3)`. The output gradient also has shape `(64, 3)`, but the bias needs a gradient of shape `(3,)`.

**Solution:** `_unbroadcast` sums the gradient along any axis that was broadcast — leading axes that don't exist in the target shape, and axes where the target size was 1. Without this, the bias gradient would be wrong in both shape and magnitude (64× too large).

---

## 4. Operations Implemented

| Operation | Forward | Backward |
|-----------|---------|----------|
| `+`, `-` | Elementwise | Unbroadcast gradient to each input |
| `*`, `/` | Elementwise | Product rule / quotient rule |
| `**` | x^n | n · x^(n-1) |
| `@` | Matrix multiply | dL/dA = dL/dC @ B.T; dL/dB = A.T @ dL/dC |
| `sum` | Sum elements | Broadcast ones back |
| `mean` | Average | Broadcast (1/N) back |
| `exp` | e^x | e^x |
| `log` | ln(x) | 1/x (clipped for stability) |
| `tanh` | tanh(x) | 1 − tanh²(x) |
| `relu` | max(0, x) | 1 where x > 0, else 0 |
| Softmax CE | Combined | softmax(z) − one_hot(y), divided by batch size |

---

## 5. Gradient Checking

### 5.1 Method

Central finite differences with ε = 1e-5:

```
f'(x) ≈ [f(x + ε) − f(x − ε)] / (2ε)
```

Relative error below `1e-4` is considered a pass.

### 5.2 Results

| Test | Status | Max Rel. Error |
|------|--------|----------------|
| Scalar: x\*y+z | PASS | ~1e-8 |
| Vector elementwise | PASS | ~1e-7 |
| Matmul with bias | PASS | ~1e-6 |
| MSE loss | PASS | ~1e-7 |
| Softmax CE | PASS | ~1e-6 |
| Custom 1–10 | PASS | all < 1e-4 |

---

## 6. Neural Network Training

### 6.1 Architecture

```
Input(2) → Linear(2→64) → ReLU → Linear(64→64) → ReLU → Linear(64→3) → Softmax CE Loss
```

### 6.2 Dataset

300 points (100 per class), linearly inseparable — the network must learn a non-linear decision boundary.

### 6.3 Training Details

- Optimizer: SGD
- Epochs: 500, full-batch
- Weight init: He initialization (std = √(2/fan_in))

### 6.4 Results

| lr | Final Accuracy |
|----|----------------|
| 0.001 | low (too slow) |
| 0.01 | moderate |
| **0.1** | **≥ 90% (best)** |
| 1.0 | low or diverging |

### 6.5 Why lr=0.1 Worked Best

- **0.001 / 0.01** — loss decreases but too slowly; not converged after 500 epochs
- **0.1** — fast, stable convergence; hits >90%
- **1.0** — overshoots the minimum, loss is noisy or diverges

### 6.6 Weight Initialisation

- **He init** — keeps activation variance stable across ReLU layers; fast convergence
- **Small (0.01)** — near-zero activations, tiny gradients, very slow start
- **Large (1.0)** — gradient explosion, unstable training

---

## 7. Debugging Investigation

### Bug: Missing `zero_grad()`

**What was wrong:** gradients accumulated across every step instead of being reset before each backward pass.

**Effect:** at step k, the effective gradient is the sum of gradients from steps 1 through k. The effective learning rate grows as k × lr and updates blow up.

**How it was detected:**
1. Loss curve didn't decrease properly
2. Accuracy stayed near random chance (~33%)
3. Gradient norm grew **linearly** across epochs — clear sign of accumulation

**Fix:** adding `optimizer.zero_grad()` before `loss.backward()` stabilized gradient norms, smoothed the loss curve, and brought accuracy from ~33% to ≥90%.

**Evidence:** `results/debugging_evidence.png` — 3-panel plot showing loss, accuracy, and gradient norm for broken vs fixed training.

---

## 8. Answers to Conceptual Questions

**Q1. Why is topological ordering necessary?**
Each node needs gradients from all downstream nodes first. Topological order guarantees that before we process any node, everything after it in the graph is already done.

**Q2. Why accumulate gradients instead of overwrite?**
A tensor used in multiple operations receives gradient contributions from each path. The correct total is their sum — overwriting would silently drop all but the last.

**Q3. What goes wrong without softmax numerical stabilization?**
Large logits cause `exp(logit)` to overflow to `inf`, producing `inf/inf = NaN`. Subtracting the row maximum before `exp` prevents this without changing the result mathematically.

**Q4. Why does broadcasting complicate the backward pass?**
The gradient arrives in the larger broadcast shape, but must be returned in the original smaller shape. You have to identify which axes were broadcast and sum over them — requiring careful shape tracking.

**Q5. Why does dL/dW have shape (input_dim, output_dim)?**
Gradients must match parameter shapes. Working through the math: dL/dW = X.T @ dL/dy = (input_dim, batch) @ (batch, output_dim) = (input_dim, output_dim). ✓

**Q6. Parameter vs activation vs gradient?**
- **Parameter** — learnable weight/bias, persists between steps, updated by optimizer
- **Activation** — layer output during forward pass, recomputed each step
- **Gradient** — derivative of loss w.r.t. a parameter or activation; drives weight updates and backpropagation

**Q7. Why might training fail even with correct gradients?**
Learning rate too large or too small, bad weight initialization (vanishing/exploding gradients), missing `zero_grad()`, dead ReLU neurons, or wrong loss function for the task.

**Q8. How does finite-difference checking detect autograd bugs?**
It estimates each gradient using only function evaluations, independently of any backward rules. If analytical and numerical gradients disagree by more than ~1e-4, the backward implementation is wrong. Testing ops in isolation pinpoints exactly which rule is broken.

**Q9. Why compare several learning rates?**
There is no universal best lr — it depends on the problem and architecture. Sweeping values reveals what range works, how sensitive training is, and which setting converges both fast and stably.

**Q10. What evidence shows the model learned a non-linear boundary?**
The decision boundary plot shows curved regions matching the spiral arms. A linear model scores ~33% on this dataset. Reaching >90% is only possible with a non-linear boundary, which the two ReLU hidden layers provide.

---

## 9. Repository Structure

```
student-autograd-project/
├── README.md
├── requirements.txt
├── src/
│   ├── tensor.py              ← Tensor class, all backward rules
│   ├── operations.py          ← functional API
│   ├── neural_network.py      ← Linear, MLP, SGD
│   └── losses.py              ← softmax_cross_entropy, MSE
├── tests/
│   ├── test_basic_ops.py
│   ├── test_matmul.py
│   ├── test_broadcasting.py
│   └── test_gradient_checking.py
├── experiments/
│   ├── generate_spiral.py
│   ├── train_model.py
│   └── debug_broken_training.py
├── results/
└── report/
    └── final_report.md
```

---

## 10. AI Use Statement

Claude was used as a coding assistant during this project, primarily for the following:

Visualization code — generating matplotlib plotting routines (ax-based figures for loss curves, accuracy curves, decision boundaries, and the debugging evidence plot), since these involved repetitive boilerplate that was tedious to write by hand
Debugging test errors — in several places in the test suite where I encountered unexpected failures, I used AI to help identify the issue; after each fix I made sure I understood the root cause so I could handle similar problems independently going forward

All core logic — the Tensor backward rules, broadcasting handling, the training loop, and the MLP implementation — was written and understood by me. The conceptual explanations, experimental design, and analysis in this report represent my own understanding. I am prepared to explain any part of the implementation in a live discussion.

---
*End of report.*
