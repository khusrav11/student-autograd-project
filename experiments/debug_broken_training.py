import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.tensor import Tensor
from src.neural_network import MLP, SGD
from src.losses import softmax_cross_entropy_loss
from experiments.generate_spiral import make_spiral
from experiments.train_model import compute_accuracy, plot_decision_boundary

# broken training (no zero_grad)
def train_broken(X_np, y_np, learning_rate=0.1, n_epochs=200, seed=42):
    model = MLP(input_dim=2, hidden_dim=64, output_dim=3, seed=seed)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    loss_hist, acc_hist, grad_norms = [], [], []
    for epoch in range(1, n_epochs + 1):
        X_t = Tensor(X_np, requires_grad=False)
        logits = model(X_t)
        loss = softmax_cross_entropy_loss(logits, y_np)
        # bug: zero_grad() is not called here
        # optimizer.zero_grad() this line is missing
        loss.backward()
        optimizer.step()
        loss_val = loss.data.item()
        acc = compute_accuracy(model, X_np, y_np)
        # track gradient norm to show explosion
        total_gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_gnorm += np.sum(p.grad ** 2)
        grad_norms.append(np.sqrt(total_gnorm))
        loss_hist.append(loss_val if np.isfinite(loss_val) else np.nan)
        acc_hist.append(acc)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  [BROKEN] Epoch {epoch:4d} | loss={loss_val:.4f} | "
                  f"acc={acc:.1f}% | grad_norm={grad_norms[-1]:.2e}")

    return model, loss_hist, acc_hist, grad_norms

# fixed training (with zero_grad)
def train_fixed(X_np, y_np, learning_rate=0.1, n_epochs=200, seed=42):
    model = MLP(input_dim=2, hidden_dim=64, output_dim=3, seed=seed)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    loss_hist, acc_hist, grad_norms = [], [], []

    for epoch in range(1, n_epochs + 1):
        X_t = Tensor(X_np, requires_grad=False)
        logits = model(X_t)
        loss = softmax_cross_entropy_loss(logits, y_np)
        # fix: zero gradients before backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.data.item()
        acc = compute_accuracy(model, X_np, y_np)
        total_gnorm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                total_gnorm += np.sum(p.grad ** 2)
        grad_norms.append(np.sqrt(total_gnorm))
        loss_hist.append(loss_val)
        acc_hist.append(acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [FIXED]  Epoch {epoch:4d} | loss={loss_val:.4f} | "
                  f"acc={acc:.1f}% | grad_norm={grad_norms[-1]:.2e}")

    return model, loss_hist, acc_hist, grad_norms
# AI work:
# produce evidence plot
def plot_debugging_evidence(broken_loss, fixed_loss, broken_acc, fixed_acc, broken_gnorms, fixed_gnorms, save_path,):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # loss
    ax = axes[0]
    ax.plot(broken_loss, color="#e74c3c", linewidth=1.5, label="Broken (no zero_grad)")
    ax.plot(fixed_loss,  color="#27ae60", linewidth=1.5, label="Fixed (zero_grad)")
    ax.set_title("Loss curve comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    ax.grid(alpha=0.3)
    # clip y-axis to see detail (broken loss can be huge)
    ymax = min(max(max(fixed_loss), 1.0), 5.0)
    ax.set_ylim(0, ymax)
    ax.annotate("Broken loss stays high\nor diverges",
                xy=(len(broken_loss) // 2, broken_loss[len(broken_loss) // 2]),
                xytext=(0.4, 0.8), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontsize=8)
    # accuracy
    ax = axes[1]
    ax.plot(broken_acc, color="#e74c3c", linewidth=1.5, label="Broken")
    ax.plot(fixed_acc,  color="#27ae60", linewidth=1.5, label="Fixed")
    ax.set_title("Accuracy comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    # gradient norms — key evidence of the bug
    ax = axes[2]
    ax.plot(broken_gnorms, color="#e74c3c", linewidth=1.5, label="Broken (accumulating)")
    ax.plot(fixed_gnorms,  color="#27ae60", linewidth=1.5, label="Fixed (stable)")
    ax.set_title("Gradient norm (evidence of bug)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||grad||")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.annotate("Gradient norm grows\nlinearly (accumulation!)",
                xy=(len(broken_gnorms) - 1, broken_gnorms[-1]),
                xytext=(0.3, 0.7), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontsize=8)
    plt.suptitle("Debugging: Missing zero_grad() causes gradient accumulation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# main
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    X, y = make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=42)
    N_EPOCHS = 200
    print("=" * 60)
    print("BUG INVESTIGATION: Missing zero_grad()")
    print("=" * 60)

    # run broken version 
    print("\n[1] Running BROKEN training (no zero_grad)...")
    broken_model, broken_loss, broken_acc, broken_gnorms = train_broken(X, y, learning_rate=0.1, n_epochs=N_EPOCHS)
    broken_final_acc = broken_acc[-1]
    print(f"\nBroken final accuracy: {broken_final_acc:.1f}%")
    print(f"Broken final gradient norm: {broken_gnorms[-1]:.2e}")

    # run fixed version 
    print("\n[2] Running FIXED training (with zero_grad)...")
    fixed_model, fixed_loss, fixed_acc, fixed_gnorms = train_fixed(X, y, learning_rate=0.1, n_epochs=N_EPOCHS)
    fixed_final_acc = fixed_acc[-1]
    print(f"\nFixed final accuracy: {fixed_final_acc:.1f}%")
    print(f"Fixed final gradient norm: {fixed_gnorms[-1]:.2e}")

    # evidence plot 
    print("\n[3] Generating debugging evidence plot...")
    plot_debugging_evidence(broken_loss, fixed_loss, broken_acc,  fixed_acc, broken_gnorms, fixed_gnorms, save_path="results/debugging_evidence.png",)

    # decision boundaries for before/after
    plot_decision_boundary(broken_model, X, y, f"Broken model (no zero_grad) — acc={broken_final_acc:.0f}%", "results/debug_broken_boundary.png")
    plot_decision_boundary(fixed_model, X, y, f"Fixed model (with zero_grad) — acc={fixed_final_acc:.0f}%", "results/debug_fixed_boundary.png")

    # written summary 
    print("\n" + "=" * 60)
    print("DEBUGGING SUMMARY")
    print("=" * 60)
    print(f"""
Bug:      Missing optimizer.zero_grad() before loss.backward()
Symptom:  - Loss does not decrease properly
          - Final accuracy is ~{broken_final_acc:.0f}% (vs ~{fixed_final_acc:.0f}% when fixed)
          - Gradient norm GROWS with epoch number (linear growth)
            because grad at step k = sum(gradients from steps 1..k)

Detection:
  1. Plotted gradient norm over epochs → saw monotonically increasing norm
  2. Effective step size is lr * k * true_grad, which blows up

Proof of fix:
  - After adding zero_grad(), gradient norm is stable
  - Final accuracy improves from ~{broken_final_acc:.0f}% to ~{fixed_final_acc:.0f}%
  - Loss curve is smooth and decreasing

Evidence: results/debugging_evidence.png
""")