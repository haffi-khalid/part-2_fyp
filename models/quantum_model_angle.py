# File: quantum_model_angle.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Purpose:
#     This file defines the quantum classifier model for angle encoding.
#     The circuit architecture is designed based on the model discussed in 
#     "Quantum Adversarial Machine Learning" (Sirui Lu et al., 2020) and is 
#     optimized for high accuracy.

#     The model circuit consists of the following steps:
#     1. Input Angle Encoding:
#          - Each classical input feature (assumed to be normalized to [0, π])
#            is encoded into a qubit using RY rotations.
#     2. Variational Circuit Layers:
#          - The circuit uses a parameterized variational ansatz with 'p' layers.
#          - Each layer comprises:
#               a) A rotation unit: Applies a single-qubit rotation (using qml.Rot)
#                  on each qubit. This gate implements arbitrary single-qubit rotations
#                  via three Euler angles.
#               b) An entangling unit: Applies a series of CNOT gates in a chain-like
#                  fashion across the qubits to create inter-qubit entanglement.
#     3. Measurement:
#          - The final output is obtained by measuring the expectation value
#            of PauliZ on the first qubit, which serves as the classifier output.

# Inputs:
#     - weights: A 2D numpy array (or list) of shape (p, 3*n_qubits) where each row 
#       corresponds to the three Euler angles for each qubit in one rotation layer.
#     - x: A 1D numpy array of length n_qubits. The input features have been normalized 
#          to [0, π] for angle encoding.

# Outputs:
#     - The circuit outputs the expectation value of PauliZ on qubit 0.
    
# Methodology:
#     The classifier uses a hardware-efficient ansatz with multiple parameterized layers.
#     Each rotation unit is implemented using qml.Rot, which is decomposed into Z, Y, and Z rotations.
#     Following each rotation layer, a fixed CNOT entangler layer is applied to induce entanglement.
#     The final measurement on qubit 0 is used as the classifier output (which can be post-processed 
#     for binary classification tasks).

# Note:
#     This model file is intended to be imported by the training script. All training-related code 
#     is maintained separately.

# Usage Example:
#     from quantum_model_angle import quantum_classifier
#     output = quantum_classifier(weights, x)

import pennylane as qml
from pennylane import numpy as np

# Number of qubits for the classifier (should match the dimensionality of input x)
n_qubits = 4  # Adjust this value as needed

# Depth (number of layers) of the variational ansatz
p = 10  # For example, p = 3 layers; can be tuned for higher accuracy

# Initialize the quantum device using the default.qubit simulator.
dev = qml.device("default.qubit", wires=n_qubits)

def angle_input_encoding(x):
    """
    Encodes classical input data into quantum states using angle encoding.
    
    Each element x[i] of the input array x (assumed in [0, π])
    is encoded into the i-th qubit via an RY rotation.
    
    Args:
        x (array): A 1D numpy array of length n_qubits.
    """
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

def rotation_layer(layer_params):
    """
    Applies a rotation layer to all qubits.
    
    Each qubit receives a parameterized arbitrary rotation via
    qml.Rot, which implements a rotation using three Euler angles.
    
    Args:
        layer_params (array): Array of three angles for each qubit.
                                Expected shape: (n_qubits, 3).
    """
    for i in range(n_qubits):
        # qml.Rot applies a rotation with three parameters (phi, theta, omega)
        qml.Rot(layer_params[i, 0], layer_params[i, 1], layer_params[i, 2], wires=i)

def entangling_layer():
    """
    Applies an entangling layer using CNOT gates in a chain-like configuration.
    
    Connects qubits in a ring or linear chain to induce entanglement.
    Here, we apply CNOT from qubit i to qubit (i+1) mod n_qubits.
    """
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i+1) % n_qubits])

@qml.qnode(dev, interface="autograd")
def quantum_classifier(weights, x):
    """
    Quantum classifier circuit for angle encoding.
    
    The circuit first encodes the input data x via angle encoding and then
    applies p layers of variational ansatz. Each layer consists of a rotation unit 
    (with arbitrary single-qubit rotations) followed by an entangling layer.
    The classifier output is obtained by measuring the expectation value of
    PauliZ on qubit 0.
    
    Args:
        weights (array): A 2D array of shape (p, n_qubits, 3). Each slice weights[i]
                         corresponds to the parameters of the rotation layer in the i-th layer.
        x (array): Input feature vector of length n_qubits, normalized to [0, π].
    
    Returns:
        float: Expectation value of PauliZ measured on qubit 0.
    """
    # Step 1: Input encoding - map classical data to quantum state via RY gates.
    angle_input_encoding(x)
    
    # Step 2: Apply p layers of the variational circuit
    for i in range(p):
        # Apply rotation layer for layer i.
        rotation_layer(weights[i])
        # Apply entangling layer.
        entangling_layer()
        
    # Step 3: Measurement: Output the expectation value of PauliZ on qubit 0.
    return qml.expval(qml.PauliZ(0))

# For testing: (This test code can be removed when integrating in the overall project)
if __name__ == "__main__":
    # Example input: x should be an array of length n_qubits
    x_example = np.array([0.5*np.pi, 0.3*np.pi, 0.7*np.pi, 0.1*np.pi])
    # Initialize random weights for p layers; each layer has shape (n_qubits, 3)
    weights_example = np.random.uniform(0, 2*np.pi, (p, n_qubits, 3))
    output_value = quantum_classifier(weights_example, x_example)
    print("Quantum classifier output (PauliZ expectation on qubit 0):", output_value)
