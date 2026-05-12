# 🌌 Quantum-AI Sentinel: Hybrid Neural Intelligence

**Quantum-AI Sentinel** is a sophisticated hybrid neural network engine that integrates classical deep learning architectures with quantum computing principles. This project bridges the gap between the linear processing power of classical CPUs and the high-dimensional **Hilbert Space** advantages of Quantum Processing Units (QPUs).

---

## 🚀 Project Vision

Modern AI models often require massive datasets to generalize. **Quantum-AI Sentinel** aims to explore the "Quantum Advantage" by encoding data into quantum states (qubits), potentially capturing complex correlations with significantly less data. This system is built upon a low-level classical engine developed **from scratch**, integrated with variational quantum circuits.

---

## 🛠️ Technical Architecture

The system is composed of three distinct architectural pillars:

### 1. The Linear Engine (Classical Foundation)

The mathematical heart of the AI. Developed using only `NumPy` to master the underlying mechanics of deep learning:

* **Manual Backpropagation:** Implementation of the Chain Rule for gradient distribution without high-level frameworks.
* **Matrix Calculus:** Efficient management of $Z = W \cdot X + b$ operations and transposed matrix alignments.
* **Custom Activations:** Optimized implementations of ReLU and Sigmoid functions and their respective derivatives.

### 2. Quantum Logic Lab (Q-Processing)

A quantum processing layer designed using the **PennyLane** framework:

* **Variational Quantum Circuits (VQC):** Trainable quantum gates that act as neural layers.
* **Entanglement:** Utilization of CNOT gates to create non-classical correlations between qubits.
* **Measurement:** Collapsing quantum states into classical probabilities for decision-making.

### 3. The Hybrid Bridge

The "translator" layer where classical data is mapped into the quantum realm:

* **Angle Encoding:** Converting classical floating-point values into `RY` and `RX` rotation gate angles.

---

## 📊 Comparative Analysis

Initial benchmarks demonstrate the convergence characteristics of the Hybrid Sentinel compared to a standard classical baseline:

| Feature | Classical Baseline | Hybrid Sentinel (Quantum) |
| --- | --- | --- |
| **Parameter Efficiency** | Medium | High |
| **Data Requirements** | High | Low (Quantum Advantage) |
| **Learning Rate Stability** | Standard | High Convergence |
| **State Space** | Linear | Exponential (Hilbert) |

---

## 💻 Installation & Usage

### Prerequisites

* Python 3.9+
* NumPy
* PennyLane
* Matplotlib (for visualization)

### Setup

```bash
git clone https://github.com/tahaerdem/Quantum-AI-Sentinel.git
cd Quantum-AI-Sentinel
pip install -r requirements.txt
python final_model.py

```

---

## 🧠 Developer's Note
---empty for now will be edited after the end of the project
 

---

## 🌟 Roadmap

* [ ] Integration with real quantum hardware (IBM Quantum / Rigetti).
* [ ] Implementation of Multi-qubit Amplitude Encoding for higher data density.
* [ ] Comparative study on financial time-series prediction vs. medical diagnostics.
