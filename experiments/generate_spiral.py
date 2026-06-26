import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

def make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=42):
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)

    for class_id in range(n_classes):
        ix = range(n_per_class * class_id, n_per_class * (class_id + 1))
        # radius goes from 0 to 1 linearly
        r = np.linspace(0.0, 1.0, n_per_class)
        # angle sweeps 4 radians for each class, offset by class_id
        t = np.linspace(class_id * 4, (class_id + 1) * 4, n_per_class)
        t = t + rng.normal(0.0, noise, n_per_class)  # add noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_id

    return X, y

def plot_spiral(X, y, save_path=None, title="Spiral Dataset"):
    colours = ["#e74c3c", "#2ecc71", "#3498db"]
    class_names = ["Class 0", "Class 1", "Class 2"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colours[cls], label=class_names[cls], s=20, alpha=0.8, edgecolors="none")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved spiral plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    X, y = make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=42)
    print("Dataset statistics:")
    print(f"  Total samples  : {len(X)}")
    print(f"  Feature shape  : {X.shape}")
    print(f"  Classes        : {np.unique(y).tolist()}")
    print(f"  Samples/class  : {np.bincount(y).tolist()}")
    print(f"  X range        : [{X.min():.3f}, {X.max():.3f}]")
    # Save a visualization of the raw data
    os.makedirs("results", exist_ok=True)
    plot_spiral(X, y, save_path="results/spiral_dataset.png")
    print("\nRun complete.")