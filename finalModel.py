"""
final_model.py
==============
Quantum-AI Sentinel — Hybrid Neural Intelligence Engine
=======================================================

Architecture (end-to-end):

    Input (classical features)
        │
        ▼
    ┌─────────────────────────────┐
    │  Classical Pre-Network      │  DenseLayer(relu) × 2
    │  (Linear Engine, scratch)   │  Learns feature representations
    └─────────────────────────────┘
        │
        ▼  (encoded to n_qubits dims)
    ┌─────────────────────────────┐
    │  Quantum Layer (VQC)        │  Angle Encoding → Variational Circuit
    │  (PennyLane)                │  → ⟨Z⟩ measurements
    └─────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────┐
    │  Classical Post-Network     │  DenseLayer(relu) → DenseLayer(sigmoid)
    │  (Linear Engine, scratch)   │  Final classification/regression head
    └─────────────────────────────┘
        │
        ▼
    Output (predictions)

Author: Taha Erdem Ersözlü
Project: Quantum-AI Sentinel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from core.linear_engine import ClassicalNet, DenseLayer, mse_loss, sigmoid
from quantum.q_circuit import QuantumLayer
from quantum.encoding import preprocess_for_quantum


# ══════════════════════════════════════════════════════
#  HYBRID SENTINEL MODEL
# ══════════════════════════════════════════════════════

class HybridSentinel:
    """
    Hybrid Classical-Quantum Neural Network.

    Workflow per forward pass:
        1. Pre-net   : classical DenseLayers compress input to n_qubits dims
        2. Encode    : classical values → qubit rotation angles
        3. Q-layer   : VQC processes encoded state, returns ⟨Z⟩ expectation values
        4. Post-net  : classical DenseLayers map Q-output to final prediction
    """

    def __init__(self,
                 input_dim: int,
                 n_qubits: int = 4,
                 q_layers: int = 2,
                 lr_classical: float = 0.05,
                 lr_quantum: float = 0.01):

        self.n_qubits = n_qubits
        self.lr_classical = lr_classical
        self.lr_quantum = lr_quantum

        # ── Pre-network: input_dim → n_qubits ──────────────────
        self.pre_layers = [
            DenseLayer(input_dim, max(input_dim, n_qubits * 2), activation="relu", seed=1),
            DenseLayer(max(input_dim, n_qubits * 2), n_qubits, activation="sigmoid", seed=2),
        ]

        # ── Quantum Layer ───────────────────────────────────────
        self.q_layer = QuantumLayer(n_qubits=n_qubits, n_layers=q_layers, seed=42)

        # ── Post-network: n_qubits → 1 ─────────────────────────
        self.post_layers = [
            DenseLayer(n_qubits, n_qubits * 2, activation="relu", seed=3),
            DenseLayer(n_qubits * 2, 1, activation="sigmoid", seed=4),
        ]

        self.loss_history = []

    # ── Forward ────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X shape: (input_dim, m)"""
        # Classical pre-network
        A = X
        for layer in self.pre_layers:
            A = layer.forward(A)      # → (n_qubits, m)

        # Encode to [0,1] for angle encoding
        A_enc = preprocess_for_quantum(A, self.n_qubits)  # → (n_qubits, m)

        # Quantum layer: ⟨Z⟩ ∈ [-1,1]^n_qubits
        Q_out = self.q_layer.forward(A_enc)               # → (n_qubits, m)

        # Normalize Q output to [0,1] for classical post-net
        Q_norm = (Q_out + 1) / 2

        # Classical post-network
        A2 = Q_norm
        for layer in self.post_layers:
            A2 = layer.forward(A2)    # → (1, m)

        return A2

    # ── Backward ───────────────────────────────────────────────

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Full backpropagation through post-net → Q-layer → pre-net."""
        m = y_true.shape[1]

        # Loss gradient at output (MSE)
        dA = 2 * (y_pred - y_true) / m

        # Post-network backward
        for layer in reversed(self.post_layers):
            dA = layer.backward(dA)

        # Quantum backward (parameter-shift gradients + dX)
        # Scale dA back from [0,1] normalization
        dA_q = dA / 2
        dX_q = self.q_layer.compute_gradients(
            preprocess_for_quantum(self.pre_layers[-1].A, self.n_qubits),
            dA_q
        )

        # Pre-network backward
        for layer in reversed(self.pre_layers):
            dX_q = layer.backward(dX_q)

    # ── Update ─────────────────────────────────────────────────

    def update(self):
        for layer in self.pre_layers + self.post_layers:
            layer.update(self.lr_classical)
        self.q_layer.update(self.lr_quantum)

    # ── Training Loop ──────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, verbose: bool = True, log_every: int = 5):
        self.loss_history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = mse_loss(y_pred, y)
            self.loss_history.append(loss)
            self.backward(y_pred, y)
            self.update()
            if verbose and epoch % log_every == 0:
                print(f"[HybridSentinel] Epoch {epoch:3d}/{epochs} | Loss: {loss:.6f}")
        return self.loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ══════════════════════════════════════════════════════
#  BASELINE CLASSICAL MODEL
# ══════════════════════════════════════════════════════

class BaselineClassical:
    """Pure classical neural network for head-to-head comparison."""

    def __init__(self, input_dim: int, lr: float = 0.05):
        self.net = ClassicalNet(
            layer_dims=[input_dim, 16, 8, 1],
            activations=["relu", "relu", "sigmoid"],
            lr=lr
        )

    def train(self, X, y, epochs=50, verbose=True, log_every=5):
        return self.net.train(X, y, epochs=epochs, verbose=verbose, log_every=log_every)

    def predict(self, X, threshold=0.5):
        return (self.net.predict(X) >= threshold).astype(int)

    def predict_proba(self, X):
        return self.net.predict(X)


# ══════════════════════════════════════════════════════
#  DATA PREPARATION
# ══════════════════════════════════════════════════════

def load_and_prepare_data(dataset: str = "moons", n_samples: int = 300, noise: float = 0.2):
    """
    Generate or load dataset.

    Options:
        'moons'          : sklearn make_moons (non-linear boundary)
        'classification' : sklearn make_classification (linear)
    """
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Transpose to (features, samples) convention used by our engine
    X_train = X_train.T
    X_test  = X_test.T
    y_train = y_train.reshape(1, -1).astype(float)
    y_test  = y_test.reshape(1, -1).astype(float)

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════

def plot_loss_comparison(hybrid_history, classical_history, save_path="notebooks/loss_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    colors = {"hybrid": "#58a6ff", "classical": "#f78166"}

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#30363d")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Loss curves
    axes[0].plot(hybrid_history,   color=colors["hybrid"],    linewidth=2, label="Hybrid Sentinel (Quantum)")
    axes[0].plot(classical_history, color=colors["classical"], linewidth=2, label="Classical Baseline")
    axes[0].set_title("Training Loss Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend(facecolor="#21262d", labelcolor="white")
    axes[0].grid(True, alpha=0.2, color="#30363d")

    # Convergence rate (smoothed)
    window = 5
    def smooth(x):
        return np.convolve(x, np.ones(window)/window, mode="valid")

    axes[1].plot(smooth(hybrid_history),    color=colors["hybrid"],    linewidth=2, label="Hybrid Sentinel")
    axes[1].plot(smooth(classical_history), color=colors["classical"], linewidth=2, label="Classical Baseline")
    axes[1].set_title("Smoothed Convergence", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss (smoothed)")
    axes[1].legend(facecolor="#21262d", labelcolor="white")
    axes[1].grid(True, alpha=0.2, color="#30363d")

    plt.suptitle("Quantum-AI Sentinel vs Classical Baseline", color="white",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Plot] Saved → {save_path}")
    plt.show()


def print_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    try:
        auc = roc_auc_score(y_true.flatten(), y_proba.flatten())
    except Exception:
        auc = float("nan")
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"{'─'*45}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  🌌  QUANTUM-AI SENTINEL  |  Hybrid Intelligence")
    print("═"*55 + "\n")

    # ── Data ──────────────────────────────────────────
    print("[Data] Generating 'moons' dataset …")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        dataset="moons", n_samples=300, noise=0.2
    )
    input_dim = X_train.shape[0]
    print(f"       Train: {X_train.shape[1]} samples | Test: {X_test.shape[1]} samples")
    print(f"       Features: {input_dim}\n")

    # ── Hybrid Model ──────────────────────────────────
    print("[Model] Initializing Hybrid Sentinel …")
    hybrid = HybridSentinel(
        input_dim=input_dim,
        n_qubits=4,
        q_layers=2,
        lr_classical=0.05,
        lr_quantum=0.01
    )
    print("[Train] Hybrid Sentinel training …\n")
    hybrid_history = hybrid.train(X_train, y_train, epochs=50, verbose=True, log_every=5)

    # ── Classical Baseline ────────────────────────────
    print("\n[Model] Initializing Classical Baseline …")
    baseline = BaselineClassical(input_dim=input_dim, lr=0.05)
    print("[Train] Classical Baseline training …\n")
    classical_history = baseline.train(X_train, y_train, epochs=50, verbose=True, log_every=5)

    # ── Evaluation ────────────────────────────────────
    print("\n[Eval] Evaluating on test set …")

    h_proba = hybrid.predict_proba(X_test)
    h_pred  = hybrid.predict(X_test)
    print_metrics("Hybrid Sentinel (Quantum)", y_test, h_pred, h_proba)

    c_proba = baseline.predict_proba(X_test)
    c_pred  = baseline.predict(X_test)
    print_metrics("Classical Baseline", y_test, c_pred, c_proba)

    # ── Visualization ─────────────────────────────────
    print("\n[Plot] Generating loss comparison chart …")
    plot_loss_comparison(hybrid_history, classical_history)

    # ── Circuit Diagram ───────────────────────────────
    print("\n[Circuit] Variational Quantum Circuit diagram:")
    hybrid.q_layer.circuit_diagram()

    print("\n✅  Quantum-AI Sentinel run complete.\n")


if __name__ == "__main__":
    main()