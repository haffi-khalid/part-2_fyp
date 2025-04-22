# File: quantum_model_amplitude.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Purpose:
#     Defines a quantum classifier using amplitude encoding for the Banknote dataset.
#     - Uses 2 qubits (wires 0 and 1) to encode the 4-dimensional feature vector via amplitude embedding.
#     - Prepares a separate output qubit (wire 2) in state |1> for readout.
#     - Applies p=10 layers of a variational ansatz: each layer has
#         1) A rotation unit on all 3 qubits (single-qubit Euler rotations via qml.Rot).
#         2) An entangling unit of CNOTs in a ring topology.
#     - Measures ⟨Z⟩ on the output qubit (wire 2) to produce the classifier output.

# Inputs:
#     - weights: numpy array of shape (p, 3, 3), where each layer has 3 qubits × 3 Euler angles.
#     - x: 1D numpy array of length 4 (normalized to unit norm for amplitude encoding).

# Outputs:
#     - Expectation value of PauliZ on qubit 2 (in [-1, 1]), which can be thresholded for binary classification.

# Usage:
#     from quantum_model_amplitude import quantum_classifier_amplitude
#     y_pred = quantum_classifier_amplitude(weights, x)

import pennylane as qml
from pennylane import numpy as np

# Number of qubits: 2 data qubits + 1 output qubit
n_qubits = 3
# Depth of the variational circuit
p = 10

# Set up the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

def amplitude_input_encoding(x):
    """
    Amplitude-encodes the 4-dimensional vector 'x' into qubits [0,1].
    Assumes ||x||₂ = 1. If not, qml.AmplitudeEmbedding(normalize=True) will normalize.
    """
    # Embed x into amplitudes of wires 0 and 1
    qml.AmplitudeEmbedding(features=x, wires=[0, 1], normalize=True)
    # Initialize output qubit to |1> for readout
    qml.PauliX(wires=2)

def rotation_layer(params):
    """
    Applies arbitrary single-qubit rotations on all qubits.
    Uses qml.Rot which implements Z–Y–Z Euler rotations.
    Args:
        params: array of shape (3, 3) → for each of the 3 qubits, three angles.
    """
    for q in range(n_qubits):
        phi, theta, omega = params[q]
        qml.Rot(phi, theta, omega, wires=q)

def entangling_layer():
    """
    Applies a ring of CNOTs to entangle qubits in a chain: 0→1, 1→2, 2→0.
    """
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q+1) % n_qubits])

@qml.qnode(dev, interface="autograd")
def quantum_classifier_amplitude(weights, x):
    """
    Amplitude-encoded quantum classifier.
    Args:
        weights: array of shape (p, n_qubits, 3)
        x: 1D array of length 4 (features)
    Returns:
        float: ⟨Z⟩ on qubit 2
    """
    # 1) Encode input
    amplitude_input_encoding(x)

    # 2) Apply p layers of variational ansatz
    for layer_idx in range(p):
        rotation_layer(weights[layer_idx])
        entangling_layer()

    # 3) Measurement on the output qubit (wire 2)
    return qml.expval(qml.PauliZ(2))

# Test stub (can be removed in final file)
if __name__ == "__main__":
    # Example normalized input vector of length 4
    x_test = np.array([0.5, 0.5, 0.5, 0.5])
    x_test = x_test / np.linalg.norm(x_test)  # ensure unit norm

    # Randomly initialize weights for p layers
    weights_test = np.random.uniform(0, 2*np.pi, (p, n_qubits, 3))
    output = quantum_classifier_amplitude(weights_test, x_test)
    print("Output ⟨Z⟩ on qubit 2:", output)
