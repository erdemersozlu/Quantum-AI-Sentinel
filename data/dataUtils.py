"""
data/dataUtils.py
==================
Data loading, generation, and preprocessing utilities.
Supports CSV datasets and synthetic benchmarks.

Author: Taha Erdem
Project: Quantum-AI Sentinel
"""

import os
import numpy as np
import csv


# ─────────────────────────────────────────────
#  Synthetic Dataset Generator
# ─────────────────────────────────────────────

def generate_spiral_dataset(n_samples: int = 200, noise: float = 0.1, seed: int = 42) -> tuple:
    """
    Two-class spiral dataset — a classic non-linearly separable benchmark.
    Returns X (n_samples, 2), y (n_samples,).
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    X, y = [], []

    for cls, sign in enumerate([1, -1]):
        theta = np.linspace(0, 4 * np.pi, n) + cls * np.pi
        r     = np.linspace(0.5, 1.0, n)
        x1    = r * np.cos(theta) + rng.normal(0, noise, n)
        x2    = r * np.sin(theta) + rng.normal(0, noise, n)
        X.append(np.stack([x1, x2], axis=1))
        y.append(np.full(n, cls))

    return np.vstack(X), np.concatenate(y)


def generate_xor_dataset(n_samples: int = 200, noise: float = 0.05, seed: int = 42) -> tuple:
    """XOR-like dataset: another non-linear boundary benchmark."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    X += rng.normal(0, noise, X.shape)
    return X, y


def generate_circles_dataset(n_samples: int = 200, noise: float = 0.05, seed: int = 42) -> tuple:
    """Concentric circles benchmark."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    X, y = [], []
    for i, r in enumerate([0.3, 0.8]):
        theta = rng.uniform(0, 2 * np.pi, n)
        x1 = r * np.cos(theta) + rng.normal(0, noise, n)
        x2 = r * np.sin(theta) + rng.normal(0, noise, n)
        X.append(np.stack([x1, x2], axis=1))
        y.append(np.full(n, i))
    return np.vstack(X), np.concatenate(y)


# ─────────────────────────────────────────────
#  CSV Dataset Generation (real-world style)
# ─────────────────────────────────────────────

def generate_sample_csv(filepath: str = "data/sample_dataset.csv",
                         n_samples: int = 500,
                         n_features: int = 4):
    """
    Generate a synthetic CSV dataset resembling a real-world binary classification task.
    Headers: feature_0, feature_1, ..., feature_n, label
    """
    rng = np.random.default_rng(42)
    rows = []
    header = [f"feature_{i}" for i in range(n_features)] + ["label"]
    rows.append(header)

    for _ in range(n_samples):
        label = rng.integers(0, 2)
        # Class 1: slightly higher mean for first two features
        means = [0.5 * label] * n_features
        features = rng.normal(means, 0.4)
        rows.append([f"{v:.4f}" for v in features] + [str(label)])

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[Data] Sample CSV saved → {filepath}  ({n_samples} rows, {n_features} features)")


# ─────────────────────────────────────────────
#  CSV Loader
# ─────────────────────────────────────────────

def load_csv(filepath: str, label_col: str = "label") -> tuple:
    """
    Load a CSV dataset.

    Returns:
        X: np.ndarray shape (n_samples, n_features)
        y: np.ndarray shape (n_samples,)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    feature_cols = [k for k in rows[0].keys() if k != label_col]
    X = np.array([[float(row[c]) for c in feature_cols] for row in rows])
    y = np.array([int(row[label_col]) for row in rows])
    return X, y


# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────

def min_max_normalize(X: np.ndarray) -> tuple:
    """Normalize each feature to [0, 1]. Returns (X_norm, min, max)."""
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    X_norm = (X - x_min) / (x_max - x_min + 1e-9)
    return X_norm, x_min, x_max


def train_test_split_manual(X: np.ndarray, y: np.ndarray,
                             test_ratio: float = 0.2, seed: int = 42):
    """Simple train/test split without sklearn dependency."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    split = int(len(y) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic datasets …")

    X_spiral, y_spiral = generate_spiral_dataset(200)
    print(f"  Spiral  : X={X_spiral.shape}, y={y_spiral.shape}, classes={np.unique(y_spiral)}")

    X_xor, y_xor = generate_xor_dataset(200)
    print(f"  XOR     : X={X_xor.shape}, y={y_xor.shape}")

    X_circ, y_circ = generate_circles_dataset(200)
    print(f"  Circles : X={X_circ.shape}, y={y_circ.shape}")

    generate_sample_csv("data/sample_dataset.csv", n_samples=500, n_features=4)
    X_csv, y_csv = load_csv("data/sample_dataset.csv")
    print(f"  CSV     : X={X_csv.shape}, y={y_csv.shape}")