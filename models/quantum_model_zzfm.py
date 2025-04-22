# File: quantum_model_zzfm.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

"""
Quantum Classifier with ZZ-Feature Map Embedding

This module defines a variational quantum classifier using:
  - A ZZ‐feature map embedding on 4 data qubits,
  - A dedicated readout qubit,
  - p = 10 layers of a hardware‐efficient ansatz (Rot + ring CNOTs),
  - Final measurement on the readout qubit.

Usage:
    from quantum_model_zzfm import quantum_classifier_zzfm
    y_pred = quantum_classifier_zzfm(weights, x)
"""

import pennylane as qml
from pennylane import numpy as np

# Total qubits: 4 data qubits + 1 readout qubit
n_qubits = 5
# Variational depth
p = 20

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

def zz_feature_map(x):
    """Embed 4‑dimensional vector x into qubits 0–3 via a simplified ZZ‐feature map."""
    # 1) Single‑qubit RZ embeddings on qubits 0–3
    for i in range(4):
        qml.RZ(x[i], wires=i)

    # 2) Nearest‑neighbor controlled‑ZZ rotations only
    neighbor_pairs = [(0, 1), (1, 2), (2, 3)]
    for i, j in neighbor_pairs:
        angle = 2 * x[i] * x[j]
        qml.MultiRZ(angle, wires=[i, j])

    # 3) Initialize the readout qubit (qubit 4) to |1⟩
    qml.PauliX(wires=4)


def rotation_layer(params):
    """Apply arbitrary single‐qubit Euler rotations on all 5 qubits."""
    for q in range(n_qubits):
        phi, theta, omega = params[q]
        qml.Rot(phi, theta, omega, wires=q)

def entangling_layer():
    """Entangle all qubits in a ring: i→(i+1) mod n_qubits."""
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])

@qml.qnode(dev, interface="autograd")
def quantum_classifier_zzfm(weights, x):
    """
    Quantum classifier with ZZ‐feature map embedding.

    Args:
        weights (array): shape (p, n_qubits, 3) of Euler angles.
        x       (array): length‑4 feature vector (scaled to [0, π]).

    Returns:
        float: ⟨Z⟩ expectation on qubit 4 (readout qubit).
    """
    # Data embedding
    zz_feature_map(x)

    # Variational ansatz
    for layer in range(p):
        rotation_layer(weights[layer])
        entangling_layer()

    # Measurement on readout qubit
    return qml.expval(qml.PauliZ(4))


# Quick test stub (remove for integration)
if __name__ == "__main__":
    # Example input vector of length 4
    x_test = np.array([0.2 * np.pi, 0.4 * np.pi, 0.6 * np.pi, 0.8 * np.pi])
    # Random weights: (p, 5, 3)
    weights_test = np.random.uniform(0, 2*np.pi, (p, n_qubits, 3))
    print("⟨Z⟩ on readout qubit:", quantum_classifier_zzfm(weights_test, x_test))
