"""
quantum/qcircuit.py
====================
Variational Quantum Circuit (VQC) Layer — built with PennyLane.
Acts as a trainable quantum "layer" that can be plugged into the hybrid model.

Architecture:
    1. Angle Encoding  → classical floats  ─→  qubit rotation angles
    2. Variational Gates → trainable RY/RZ rotations + CNOT entanglement
    3. Measurement     → expectation values ⟨Z⟩ per qubit → classical vector

Author: Taha Erdem Ersözlü
Project: Quantum-AI Sentinel
"""

import numpy as np
import pennylane as qml


# ─────────────────────────────────────────────
#  Device & Circuit Definition
# ─────────────────────────────────────────────

class QuantumLayer:
    """
    A single Variational Quantum Circuit layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= feature dimension after encoding).
    n_layers : int
        Depth of the variational ansatz (more layers → more expressibility).
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, seed: int = 0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # PennyLane simulator device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Trainable parameters: shape (n_layers, n_qubits, 2)
        # Each qubit gets an RY and an RZ rotation per layer
        rng = np.random.default_rng(seed)
        self.params = rng.uniform(-np.pi, np.pi, (n_layers, n_qubits, 2))

        # Build the QNode
        self._circuit = qml.QNode(self._ansatz, self.dev, interface="autograd")

        # Gradient of params (manual finite-difference)
        self.grad_params = np.zeros_like(self.params)

        # Cache
        self._last_input = None

    # ── Encoding ──────────────────────────────

    def _angle_encode(self, x: np.ndarray):
        """
        Angle Encoding: maps each classical feature xᵢ to a qubit rotation.
        RY(xᵢ) followed by RX(xᵢ) to capture two angular dimensions.
        """
        for i in range(self.n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
            qml.RX(x[i] * np.pi / 2, wires=i)

    # ── Variational Ansatz ────────────────────

    def _variational_block(self, params_layer):
        """One layer of the variational ansatz: rotations + CNOT entanglement."""
        # Single-qubit rotations
        for i in range(self.n_qubits):
            qml.RY(params_layer[i, 0], wires=i)
            qml.RZ(params_layer[i, 1], wires=i)

        # Entanglement: ring of CNOT gates
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def _ansatz(self, x, params):
        """Full circuit: encode → variational layers → measure."""
        self._angle_encode(x)
        for layer_params in params:
            self._variational_block(layer_params)
        # Return Pauli-Z expectation per qubit → vector in [-1, 1]^n_qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    # ── Forward Pass ──────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run circuit for each sample in X.

        Args:
            X: shape (n_qubits, m) — each column is one sample
        Returns:
            output: shape (n_qubits, m)
        """
        self._last_input = X
        m = X.shape[1]
        outputs = []
        for j in range(m):
            x_j = X[:, j]
            out = self._circuit(x_j, self.params)
            outputs.append(np.array(out))
        return np.stack(outputs, axis=1)   # (n_qubits, m)

    # ── Parameter Gradient (Finite Difference) ──

    def compute_gradients(self, X: np.ndarray, dA: np.ndarray, eps: float = 1e-3):
        """
        Estimate ∂L/∂params via parameter-shift rule (finite difference approximation).
        Also computes dX for chaining back to classical layers.

        Args:
            X:  shape (n_qubits, m)
            dA: upstream gradient, shape (n_qubits, m)
        """
        self.grad_params = np.zeros_like(self.params)
        m = X.shape[1]

        # Parameter-shift rule: ∂f/∂θ ≈ (f(θ+ε) - f(θ-ε)) / (2ε)
        it = np.nditer(self.params, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = self.params[idx]

            self.params[idx] = original + eps
            out_plus = self.forward(X)

            self.params[idx] = original - eps
            out_minus = self.forward(X)

            self.params[idx] = original

            grad = np.sum(dA * (out_plus - out_minus) / (2 * eps)) / m
            self.grad_params[idx] = grad
            it.iternext()

        # dX: finite difference w.r.t. inputs
        dX = np.zeros_like(X)
        for i in range(self.n_qubits):
            X_plus = X.copy();  X_plus[i, :] += eps
            X_minus = X.copy(); X_minus[i, :] -= eps
            out_plus  = self.forward(X_plus)
            out_minus = self.forward(X_minus)
            dX[i, :] = np.sum(dA * (out_plus - out_minus) / (2 * eps), axis=0)

        return dX

    def update(self, lr: float = 0.01):
        """Adam-like gradient descent on quantum params."""
        self.params -= lr * self.grad_params

    def circuit_diagram(self):
        """Print a text diagram of the circuit."""
        sample_x = np.zeros(self.n_qubits)
        print(qml.draw(self._circuit)(sample_x, self.params))