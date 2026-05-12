"""
quantum/encoding.py
===================
Classical → Quantum Encoding Strategies.

Supported encodings:
    - Angle Encoding  (default, used in VQC)
    - Amplitude Encoding (normalize to unit vector)
    - Z-score normalization helper

Author: Taha Erdem Ersözlü
Project: Quantum-AI Sentinel
"""

import numpy as np


def angle_encode(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Angle Encoding: truncate/pad x to n_qubits and scale to [0, 1].
    The VQC then multiplies by π internally for rotation angles.
    """
    x_scaled = (x - x.min()) / (x.ptp() + 1e-9)
    if len(x_scaled) >= n_qubits:
        return x_scaled[:n_qubits]
    else:
        return np.pad(x_scaled, (0, n_qubits - len(x_scaled)))


def amplitude_encode(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Amplitude Encoding: normalize x to a unit vector of length 2^n_qubits.
    Useful for dense state representations.
    """
    dim = 2 ** n_qubits
    x_padded = np.zeros(dim)
    x_padded[:min(len(x), dim)] = x[:min(len(x), dim)]
    norm = np.linalg.norm(x_padded)
    return x_padded / (norm + 1e-9)


def preprocess_for_quantum(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Prepare a classical feature matrix for quantum encoding.

    Args:
        X:         shape (features, m)  — standard column-per-sample format
        n_qubits:  number of qubits in the VQC

    Returns:
        X_enc:     shape (n_qubits, m)
    """
    m = X.shape[1]
    X_enc = np.zeros((n_qubits, m))
    for j in range(m):
        X_enc[:, j] = angle_encode(X[:, j], n_qubits)
    return X_enc


def zscore_normalize(X: np.ndarray):
    """Column-wise Z-score normalization. Returns normalized X, mean, std."""
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-9
    return (X - mu) / sigma, mu, sigma