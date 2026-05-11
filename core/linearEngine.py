"""
core/linear_engine.py
=====================
Classical Neural Network Engine — built from scratch using only NumPy.
No high-level frameworks. Pure matrix calculus and manual backpropagation.

Author: Taha Erdem Ersözlü
Project: Quantum-AI Sentinel
"""

import numpy as np


# ─────────────────────────────────────────────
#  Activation Functions & Their Derivatives
# ─────────────────────────────────────────────

def relu(Z):
    """Rectified Linear Unit activation."""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """Derivative of ReLU — 1 where Z > 0, else 0."""
    return (Z > 0).astype(float)


def sigmoid(Z):
    """Sigmoid activation — squashes output to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))


def sigmoid_derivative(Z):
    """Derivative of Sigmoid: s(z) * (1 - s(z))."""
    s = sigmoid(Z)
    return s * (1 - s)


def tanh_activation(Z):
    return np.tanh(Z)


def tanh_derivative(Z):
    return 1 - np.tanh(Z) ** 2


# ─────────────────────────────────────────────
#  Loss Functions
# ─────────────────────────────────────────────

def binary_cross_entropy(y_pred, y_true):
    """Binary Cross-Entropy loss."""
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]


# ─────────────────────────────────────────────
#  Dense Layer
# ─────────────────────────────────────────────

class DenseLayer:
    """
    A fully-connected layer: Z = W · X + b
    Supports ReLU, Sigmoid, and Tanh activations.
    """

    ACTIVATIONS = {
        "relu":    (relu,            relu_derivative),
        "sigmoid": (sigmoid,         sigmoid_derivative),
        "tanh":    (tanh_activation, tanh_derivative),
        "linear":  (lambda z: z,     lambda z: np.ones_like(z)),
    }

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu", seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialization for ReLU, Xavier for others
        if activation == "relu":
            scale = np.sqrt(2.0 / input_dim)
        else:
            scale = np.sqrt(1.0 / input_dim)

        self.W = rng.normal(0, scale, (output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        self.activation_name = activation
        self.act_fn, self.act_deriv = self.ACTIVATIONS[activation]

        # Cache for backprop
        self.X = None   # input
        self.Z = None   # pre-activation
        self.A = None   # post-activation

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass:
            Z = W · X + b
            A = activation(Z)
        """
        self.X = X
        self.Z = self.W @ X + self.b
        self.A = self.act_fn(self.Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass (Chain Rule):
            dZ = dA * activation'(Z)
            dW = dZ · Xᵀ / m
            db = mean(dZ)
            dX = Wᵀ · dZ
        """
        m = self.X.shape[1]
        dZ = dA * self.act_deriv(self.Z)
        self.dW = (dZ @ self.X.T) / m
        self.db = np.mean(dZ, axis=1, keepdims=True)
        dX = self.W.T @ dZ
        return dX

    def update(self, lr: float):
        """Gradient descent weight update."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


# ─────────────────────────────────────────────
#  Classical Neural Network
# ─────────────────────────────────────────────

class ClassicalNet:
    """
    A fully-connected neural network composed of DenseLayer objects.
    Training uses vanilla SGD with manual backpropagation.
    """

    def __init__(self, layer_dims: list, activations: list, lr: float = 0.01):
        """
        Args:
            layer_dims:  [input_dim, hidden1, hidden2, ..., output_dim]
            activations: activation names for each layer (len = len(layer_dims) - 1)
            lr:          learning rate
        """
        assert len(layer_dims) - 1 == len(activations), \
            "Number of activations must match number of layers."
        self.lr = lr
        self.layers = []
        for i in range(len(activations)):
            layer = DenseLayer(layer_dims[i], layer_dims[i + 1],
                               activation=activations[i], seed=42 + i)
            self.layers.append(layer)
        self.loss_history = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        # Initial gradient from MSE loss
        dA = mse_derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, verbose: bool = True, log_every: int = 10):
        """
        Full training loop.

        Args:
            X:        Input matrix, shape (features, samples)
            y:        Target matrix, shape (output_dim, samples)
            epochs:   Number of training epochs
            verbose:  Print loss every `log_every` epochs
        """
        self.loss_history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = mse_loss(y_pred, y)
            self.loss_history.append(loss)
            self.backward(y_pred, y)
            self.update()

            if verbose and epoch % log_every == 0:
                print(f"[ClassicalNet] Epoch {epoch:4d}/{epochs} | Loss: {loss:.6f}")

        return self.loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)