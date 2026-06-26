# Mini Automatic Differentiation Engine

A NumPy-based automatic differentiation (autograd) engine built **from scratch** — no PyTorch, no TensorFlow. Uses the engine to train a two-layer MLP on a 3-class spiral dataset achieving **98.7% accuracy**.

## What This Project Does

1. **Implements a `Tensor` class** with full computation graph and reverse-mode backpropagation
2. **Supports all required operations**: add, sub, mul, div, pow, matmul, sum, mean, exp, log, tanh, relu, softmax CE
3. **Handles broadcasting** correctly in both forward and backward passes
4. **Passes all 15 gradient checks** (5 required + 10 custom finite-difference tests)
5. **Trains a 2-layer MLP** on a 3-class spiral dataset → **98.7% accuracy**
6. **Runs hyperparameter experiments** (learning rate + weight init)
7. **Debugs a broken training run** (missing `zero_grad()` bug)

## Installation

```bash
git clone <>
cd student-autograd-project
pip install -r requirements.txt
```

**Dependencies** (only standard scientific Python — no deep learning frameworks):
```
numpy
matplotlib
```

## Running the Tests

```bash
# Run all tests
python3 -m unittest discover tests/ -v

# Run individual test files
python3 tests/test_basic_ops.py
python3 tests/test_matmul.py
python3 tests/test_broadcasting.py
python3 tests/test_gradient_checking.py
```

**Expected output:** all tests pass (25 basic ops, 11 matmul, 9 broadcasting, 15 gradient checks = 60 total tests).

## Generating the Dataset

```bash
python3 experiments/generate_spiral.py
# Saves: results/spiral_dataset.png
```

## Training the Model

```bash
python3 experiments/train_model.py
```

This trains the default model and runs all hyperparameter experiments. Saves all plots to `results/`.

**Expected output:**
```
Final loss     : 0.1068
Final accuracy : 98.7%
✓ Target of 90% accuracy reached!
```

## Debugging Investigation

```bash
python3 experiments/debug_broken_training.py
# Saves: results/debugging_evidence.png
```

## Saving Gradient Check Report

```bash
python3 tests/test_gradient_checking.py
# Also saves: results/gradient_check_results.txt
```

## Full Reproduction (One Command Per Step)

```bash
python3 experiments/generate_spiral.py
python3 experiments/train_model.py
python3 experiments/debug_broken_training.py
python3 tests/test_gradient_checking.py
python3 -m unittest discover tests/ -v
```

## Results Summary

| Metric | Value |
|--------|-------|
| Final training accuracy | **98.7%** |
| Final training loss | 0.107 |
| Gradient check pass rate | **15/15** |
| Best learning rate | 0.1 |
| Best weight init | He init |

## Plots in `results/`

| File | Description |
|------|-------------|
| `loss_curve.png` | Training loss over epochs |
| `accuracy_curve.png` | Training accuracy over epochs |
| `decision_boundary.png` | Learned non-linear class boundary |
| `lr_comparison.png` | Effect of lr ∈ {0.001, 0.01, 0.1, 1.0} |
| `init_comparison.png` | Effect of weight init scale |
| `debugging_evidence.png` | Before/after zero_grad() bug fix |
| `gradient_check_results.txt` | Detailed grad check report |

## Project Structure

```
student-autograd-project/
├── README.md
├── requirements.txt
├── src/
│   ├── tensor.py           ← Tensor class + all backward rules
│   ├── operations.py       ← functional API wrappers
│   ├── neural_network.py   ← Linear layer, MLP, SGD optimizer
│   └── losses.py           ← loss functions
├── tests/
│   ├── test_basic_ops.py
│   ├── test_matmul.py
│   ├── test_broadcasting.py
│   └── test_gradient_checking.py
├── experiments/
│   ├── generate_spiral.py
│   ├── train_model.py
│   └── debug_broken_training.py
├── results/                ← all saved plots and reports
└── report/
    └── final_report.md     ← full written report + conceptual Q&A
```

## Written Report

See [`report/final_report.md`](report/final_report.md) — includes:
- Full design explanation of autograd engine
- Answers to all 10 conceptual questions
- Hyperparameter analysis with evidence
- Complete debugging investigation
