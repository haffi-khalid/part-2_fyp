# File: quantum_model_reupload.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]
#
# Purpose:
#     Implements a variational quantum classifier using the data
#     re‑uploading ansatz for the Banknote Authentication dataset.
#     This model interleaves classical data encoding and trainable
#     circuit layers to achieve high expressivity on only five qubits:
#       - Wires 0–3 encode the four continuous features via RY rotations.
#       - Wire 4 serves as the readout qubit.
#     At each of p layers, the input features are “re‑uploaded” into
#     the circuit, followed by a parameterized rotation block and
#     a chain of entangling CNOT gates.  The final classification
#     score is ⟨Z⟩ on the readout qubit.
#
# Rationale:
#     We chose the data re‑uploading architecture because Pérez‑Salinas
#     et al. (2020) demonstrated that re‑injecting classical data at
#     multiple depths yields a universal quantum classifier on
#     low‑qubit hardware (see C. Pérez‑Salinas et al., “Data re‑uploading
#     for a universal quantum classifier,” Quantum 4, 226 (2020)).
#     This approach often outperforms static feature maps (e.g. ZZFM)
#     on low‑dimensional numerical datasets by allowing the circuit
#     to adaptively reshape its feature space.
#
# References:
#     • Pérez‑Salinas, C., Cervera‑Lierta, A., Gil‑Flores, F. et al.
#       Data re‑uploading for a universal quantum classifier. Quantum 4, 226 (2020).
#     • Havlíček, V., Córcoles, A. D., Temme, K. et al.
#       Supervised learning with quantum‑enhanced feature spaces.
#       Nature 567, 209–212 (2019).
#     • Schuld, M., Fingerhuth, M., Petruccione, F.
#       Implementation of quantum support vector machines.
#       Phys. Rev. A 94, 022305 (2016).
#
# Usage:
#     from quantum_model_reupload import quantum_classifier_reupload, n_qubits, p
#     score = quantum_classifier_reupload(weights, x_vector)
#     where x_vector is a length‑4 numpy array with values in [0, π].

import pennylane as qml
from pennylane import numpy as np

# Number of qubits: 4 for data + 1 for readout
n_qubits = 5

# Depth of the variational circuit (number of re‑uploading layers)
p = 10

# Initialize the quantum device (simulator)
dev = qml.device("default.qubit", wires=n_qubits)

def rotation_layer(params):
    """
    Apply a block of arbitrary single‑qubit rotations.
    Each qubit wire receives three Euler angles.
    
    Args:
        params (array[p, n_qubits, 3]): rotation parameters.
    """
    for wire in range(n_qubits):
        phi, theta, omega = params[wire]
        qml.Rot(phi, theta, omega, wires=wire)

def entangling_layer():
    """
    Apply a linear chain of CNOT gates to entangle all qubits:
    CNOT(0→1), CNOT(1→2), ..., CNOT(3→4).
    """
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

@qml.qnode(dev, interface="autograd")
def quantum_classifier_reupload(weights, x):
    """
    Quantum classifier using data re‑uploading.
    
    Args:
        weights (array[p, n_qubits, 3]): Variational parameters for each layer.
        x       (array[4]): Four‑dimensional input vector, scaled to [0, π].
    
    Returns:
        float: expectation value ⟨Z⟩ on the readout qubit (wire 4).
    """
    for layer in range(p):
        # 1) Re‑upload data: RY(x[j]) on data qubits 0–3
        for j in range(4):
            qml.RY(x[j], wires=j)
        # 2) Variational rotations on all qubits
        rotation_layer(weights[layer])
        # 3) Entangling CNOT chain
        entangling_layer()

    # Measure PauliZ on the readout qubit as the classifier output
    return qml.expval(qml.PauliZ(wires=4))

# Quick sanity check (remove or comment out in production)
if __name__ == "__main__":
    x_test = np.array([0.2*np.pi, 0.4*np.pi, 0.6*np.pi, 0.8*np.pi])
    # Randomly initialize weights for testing
    weights_test = np.random.uniform(0, 2*np.pi, (p, n_qubits, 3))
    print("Test ⟨Z⟩ output:", quantum_classifier_reupload(weights_test, x_test))
