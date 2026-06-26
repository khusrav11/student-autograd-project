import numpy as np
from src.tensor import Tensor
from src.losses import softmax_cross_entropy_loss

class Linear:
    # fully-connected layer: output = input @ W + b
    def __init__(self, in_features, out_features, init_scale=None, seed=None):
        rng = np.random.default_rng(seed)
        # He_init by default: std = sqrt(2 / in_features)
        if init_scale is None:
            std = np.sqrt(2.0 / in_features)
        else:
            std = init_scale
        # weight matrix: shape (in_features, out_features)
        # we should do requires_grad=True so gradients are computed during backward()
        self.W = Tensor(rng.normal(0.0, std, (in_features, out_features)), requires_grad=True)
        # Bias vector: shape (out_features,), initialised to zero
        self.b = Tensor(np.zeros(out_features), requires_grad=True)
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        # x: (batch_size, in_features), out: (batch_size, out_features), b-(bias) is broadcast over batch dim automatically
        return x @ self.W + self.b

    def parameters(self):
        # return list of learnable parameters
        return [self.W, self.b]

    def zero_grad(self):
        # reset gradients of all parameters
        for p in self.parameters():
            p.zero_grad() 

    def __repr__(self):
        return f"Linear({self.in_features} → {self.out_features})"
    
class MLP:
    # Two-hidden-layer MLP for spiral dataset classification
    # Architecture: Input(2) → Linear(2→H) → ReLU → Linear(H→H) → ReLU → Linear(H→3)
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=3, init_scale=None, seed=42):
        # We offset seeds per layer so they don't all get the same weights
        self.layer1 = Linear(input_dim,  hidden_dim,  init_scale, seed=seed)
        self.layer2 = Linear(hidden_dim, hidden_dim,  init_scale, seed=seed + 1)
        self.layer3 = Linear(hidden_dim, output_dim,  init_scale, seed=seed + 2)

    def forward(self, x):
        h1 = self.layer1(x).relu()   # hidden layer 1 + ReLU
        h2 = self.layer2(h1).relu()  # hidden layer 2 + ReLU
        logits = self.layer3(h2)      # output logits (no activation)
        return logits

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        # return all learnable parameters across all layers
        params = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        # reset all parameter gradients before a new training step
        for p in self.parameters():
            p.zero_grad()

    def predict(self, x_np):
        # runing forward pass on a NumPy array and return predicted class indices and doesn't require grad
        x = Tensor(x_np)
        logits = self.forward(x)
        return np.argmax(logits.data, axis=1)

    def __repr__(self):
        return (
            f"MLP(\n"
            f"  {self.layer1}\n"
            f"  ReLU\n"
            f"  {self.layer2}\n"
            f"  ReLU\n"
            f"  {self.layer3}\n"
            f")"
        )

class SGD:
    # Stochastic Gradient Descent optimizer
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        # update all parameters using their current gradients
        for p in self.parameters:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        # reset gradients of all managed parameters
        for p in self.parameters:
            p.zero_grad()