import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.tensor import Tensor
from src.neural_network import MLP, SGD
from src.losses import softmax_cross_entropy_loss
from experiments.generate_spiral import make_spiral, plot_spiral

# utility functions
def compute_accuracy(model, X_np, y_np):
    # compute classification accuracy (0–100%)
    preds = model.predict(X_np)
    return 100.0 * np.mean(preds == y_np)

def plot_loss_accuracy(loss_hist, acc_hist, title, save_path):
    # save loss + accuracy curves to a single PNG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(loss_hist, color="#e74c3c", linewidth=1.5)
    ax1.set_title(f"Loss — {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(alpha=0.3)

    ax2.plot(acc_hist, color="#2ecc71", linewidth=1.5)
    ax2.set_title(f"Accuracy — {title}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

def plot_decision_boundary(model, X_np, y_np, title, save_path, h=0.02):
    x_min, x_max = X_np[:, 0].min() - 0.2, X_np[:, 0].max() + 0.2
    y_min, y_max = X_np[:, 1].min() - 0.2, X_np[:, 1].max() + 0.2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_bg = plt.get_cmap("Pastel1", 3)
    ax.contourf(xx, yy, preds, cmap=cmap_bg, alpha=0.6, levels=[-0.5, 0.5, 1.5, 2.5])
    colours = ["#e74c3c", "#2ecc71", "#3498db"]
    for cls in np.unique(y_np):
        mask = y_np == cls
        ax.scatter(X_np[mask, 0], X_np[mask, 1], c=colours[cls], s=15, alpha=0.8, edgecolors="none", label=f"Class {cls}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# core training function
def train(
    X_np, y_np,
    hidden_dim=64,
    learning_rate=0.1,
    n_epochs=500,
    batch_size=None,   # None → full-batch gradient descent
    init_scale=None,   # None → He init
    seed=42,
    verbose=True,
):
    """
    Train the MLP on the spiral dataset.

    Parameters:
    X_np         : ndarray (N, 2)
    y_np         : ndarray (N,)
    hidden_dim   : int
    learning_rate: float
    n_epochs     : int
    batch_size   : int or None (None = full batch)
    init_scale   : float or None (None = He init)
    seed         : int

    Returns:
    model      : trained MLP
    loss_hist  : list of float
    acc_hist   : list of float
    """
    model = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=3, init_scale=init_scale, seed=seed,)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    N = len(X_np)
    bs = N if batch_size is None else batch_size
    loss_hist, acc_hist = [], []

    for epoch in range(1, n_epochs + 1):
        # mini-batch loop (full batch when batch_size=None)
        epoch_loss = 0.0
        n_batches = 0
        idx = np.random.default_rng(seed + epoch).permutation(N)

        for start in range(0, N, bs):
            batch_idx = idx[start:start + bs]
            X_batch = Tensor(X_np[batch_idx], requires_grad=False)
            y_batch = y_np[batch_idx]
            # 1. forward pass
            logits = model(X_batch)
            # 2. compute loss
            loss = softmax_cross_entropy_loss(logits, y_batch)
            # 3. zero gradients BEFORE backward (crucial!)
            optimizer.zero_grad()
            # 4. backward pass
            loss.backward()
            # 5. gradient descent step
            optimizer.step()
            epoch_loss += loss.data.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        acc = compute_accuracy(model, X_np, y_np)
        loss_hist.append(avg_loss)
        acc_hist.append(acc)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  Epoch {epoch:4d} | loss={avg_loss:.4f} | acc={acc:.1f}%")

    return model, loss_hist, acc_hist

# experiment: learning rate comparison
def experiment_learning_rates(X_np, y_np):
    # train with 4 different learning rates and compare loss curves
    lrs = [0.001, 0.01, 0.1, 1.0]
    colours = ["#8e44ad", "#e74c3c", "#27ae60", "#e67e22"]
    n_epochs = 500
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    results = {}

    for lr, colour in zip(lrs, colours):
        print(f"\n--- LR = {lr} ---")
        model, loss_hist, acc_hist = train(X_np, y_np, learning_rate=lr, n_epochs=n_epochs, verbose=True)
        final_acc = acc_hist[-1]
        results[lr] = (model, loss_hist, acc_hist, final_acc)
        label = f"lr={lr} (final acc={final_acc:.0f}%)"
        axes[0].plot(loss_hist, label=label, color=colour, linewidth=1.5)
        axes[1].plot(acc_hist,  label=label, color=colour, linewidth=1.5)

    for ax, ylabel, title in zip(axes, ["Cross-entropy loss", "Training accuracy (%)"], ["Loss curves — learning rate comparison", "Accuracy curves — learning rate comparison"],):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    path = "results/lr_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    return results

# experiment: weight initialisation comparison
def experiment_init_scales(X_np, y_np):
    # compare small init (0.01) vs He init (None)
    configs = [
        ("He init (recommended)",    None,  "#27ae60"),
        ("Small init (scale=0.01)", 0.01,  "#e74c3c"),
        ("Large init (scale=1.0)",  1.0,   "#8e44ad"),
    ]
    n_epochs = 500
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, scale, colour in configs:
        print(f"\n--- Init scale: {label} ---")
        model, loss_hist, acc_hist = train(X_np, y_np, learning_rate=0.1, n_epochs=n_epochs, init_scale=scale, verbose=True)
        final_acc = acc_hist[-1]
        plot_label = f"{label} (final acc={final_acc:.0f}%)"
        axes[0].plot(loss_hist, label=plot_label, color=colour, linewidth=1.5)
        axes[1].plot(acc_hist,  label=plot_label, color=colour, linewidth=1.5)

    for ax, ylabel, title in zip(axes, ["Cross-entropy loss", "Training accuracy (%)"], ["Loss — init scale comparison", "Accuracy — init scale comparison"],):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    path = "results/init_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")
# AI work:
# main: default training run + all experiments
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # generate dataset
    print("=" * 55)
    print("Generating spiral dataset...")
    X, y = make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=42)
    print(f"Dataset: {X.shape[0]} samples, 2 features, 3 classes")
    # default training run
    print("\n" + "=" * 55)
    print("DEFAULT TRAINING RUN  (lr=0.1, He init, 500 epochs)")
    print("=" * 55)
    model, loss_hist, acc_hist = train(
        X, y,
        hidden_dim=64,
        learning_rate=0.1,
        n_epochs=500,
        verbose=True,
    )
    final_loss = loss_hist[-1]
    final_acc  = acc_hist[-1]
    print(f"\nFinal loss     : {final_loss:.4f}")
    print(f"Final accuracy : {final_acc:.1f}%")
    if final_acc >= 90.0:
        print("Target of 90% accuracy reached!")
    else:
        print("Did not reach 90% — see report for explanation.")
    # save main plots
    plot_loss_accuracy(loss_hist, acc_hist, "Default run (lr=0.1)", "results/loss_curve.png")
    # save separate accuracy plot (assignment requires it)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(acc_hist, color="#2ecc71", linewidth=1.5)
    ax.set_title("Training accuracy — default run")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/accuracy_curve.png", dpi=150)
    plt.close()
    print("Saved: results/accuracy_curve.png")
    # decision boundary
    plot_decision_boundary(model, X, y, f"Decision boundary (lr=0.1, acc={final_acc:.0f}%)", "results/decision_boundary.png")
    # learning rate experiments
    print("\n" + "=" * 55)
    print("LEARNING RATE EXPERIMENTS")
    print("=" * 55)
    experiment_learning_rates(X, y)
    # init_scale experiments
    print("\n" + "=" * 55)
    print("WEIGHT INIT EXPERIMENTS")
    print("=" * 55)
    experiment_init_scales(X, y)
    print("\n" + "=" * 55)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved in: results/")
    print("=" * 55)