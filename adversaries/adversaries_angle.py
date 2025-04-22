# File: quantum_adversarial_attacks.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Purpose:
#     This file implements adversarial attack methods (FGSM and BIM) for the quantum classifier model 
#     using angle encoding. The attacks are based on gradient-based methods as described in the 
#     "Quantum Adversarial Machine Learning" paper.

#     The functions provided are:
#     1. fgsm_attack: Generates an adversarial example using the Fast Gradient Sign Method (FGSM).
#     2. bim_attack: Generates an adversarial example using the Basic Iterative Method (BIM).

#     Both methods work by computing the gradient of a defined loss function with respect to the input 
#     data and perturbing the input in the direction that maximizes the loss. After each perturbation, 
#     the inputs are clipped to remain within the valid range [0, π] for angle encoding.

# Inputs:
#     - weights: The current variational parameters of the quantum model (shape: (p, n_qubits, 3)).
#     - x: The original input feature vector (1D numpy array of length n_qubits, with values in [0, π]).
#     - y: The true target label (assumed here to be a real value; for binary classification, y is typically {-1, 1}).
#     - epsilon: The magnitude of the adversarial perturbation.
#     - (For BIM) iterations: Number of iterative steps, and alpha: step size per iteration.

# Outputs:
#     - Returns the adversarial example x_adv generated from the input x.

# Usage:
#     Import these functions into your training file and use them to generate adversarial examples during 
#     adversarial training.
    
# Note:
#     This code assumes that the quantum classifier model (quantum_classifier) and necessary libraries 
#     (PennyLane, numpy) have been properly imported.

import pennylane as qml
from pennylane import numpy as np
from models.quantum_model_angle import quantum_classifier  # Ensure this file is in your PYTHONPATH

# Define the valid input range for angle encoding.
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

def cost_function(weights, x, y):
    """
    Defines the loss function for the quantum classifier.
    Here we use a simple squared error loss function:
    
        L(weights, x, y) = [quantum_classifier(weights, x) - y]^2

    Args:
        weights (array): Variational parameters for the quantum model.
        x (array): Input feature vector.
        y (float): True label for the input.
        
    Returns:
        float: Squared error loss.
    """
    prediction = quantum_classifier(weights, x)
    return (prediction - y) ** 2

def fgsm_attack(weights, x, y, epsilon):
    """
    Implements the Fast Gradient Sign Method (FGSM) for adversarial attacks on the quantum classifier.
    
    Given an input x, the adversarial example is generated as:
    
        x_adv = clip(x + epsilon * sign(∇_x L(weights, x, y)), ANGLE_MIN, ANGLE_MAX)
    
    Args:
        weights (array): The parameters of the quantum model.
        x (array): Original input feature vector.
        y (float): True label.
        epsilon (float): Magnitude of the perturbation.
        
    Returns:
        array: The adversarial input x_adv.
    """
    # Compute gradient of loss with respect to input x.
    grad_fn = qml.grad(cost_function, argnum=1)
    grad_x = grad_fn(weights, x, y)
    
    # Apply FGSM perturbation.
    x_adv = x + epsilon * np.sign(grad_x)
    
    # Clip the adversarial example to stay within [ANGLE_MIN, ANGLE_MAX].
    x_adv = np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)
    
    return x_adv

def bim_attack(weights, x, y, epsilon, iterations, alpha):
    """
    Implements the Basic Iterative Method (BIM) for generating adversarial examples.
    
    BIM is an iterative application of FGSM. At each iteration, the input is perturbed by:
    
        x_{t+1} = clip(x_t + alpha * sign(∇_x L(weights, x_t, y)), ANGLE_MIN, ANGLE_MAX)
    
    The final adversarial example is obtained after the specified number of iterations.
    
    Args:
        weights (array): The parameters of the quantum model.
        x (array): Original input feature vector.
        y (float): True label.
        epsilon (float): Maximum total perturbation allowed.
        iterations (int): Number of iterations.
        alpha (float): Step size for each iteration (suggested: alpha <= epsilon).
        
    Returns:
        array: The adversarial input x_adv after iterations.
    """
    # Initialize x_adv as the original input.
    x_adv = np.copy(x)
    
    # Perform iterative attack.
    for i in range(iterations):
        # Compute the gradient at current adversarial input.
        grad_fn = qml.grad(cost_function, argnum=1)
        grad_x = grad_fn(weights, x_adv, y)
        
        # Update adversarial input with a small step in the direction of the gradient sign.
        x_adv = x_adv + alpha * np.sign(grad_x)
        
        # Ensure the perturbation does not exceed epsilon from the original input.
        # First, clip the cumulative perturbation.
        perturbation = np.clip(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturbation
        
        # Finally, clip to valid angle range.
        x_adv = np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)
    
    return x_adv

# Test code for FGSM and BIM attacks.
if __name__ == "__main__":
    # Example input: ensure the length of x equals n_qubits defined in quantum_model_angle.py.
    # Here we assume n_qubits is 4.
    x_example = np.array([0.5 * np.pi, 0.3 * np.pi, 0.7 * np.pi, 0.1 * np.pi])
    
    # Example: random weights for p layers; shape: (p, n_qubits, 3)
    from models.quantum_model_angle import p  # p is defined in quantum_model_angle.py
    n_qubits = 4  # Ensure consistency with quantum_model_angle.py
    weights_example = np.random.uniform(0, 2 * np.pi, (p, n_qubits, 3))
    
    # True label (for example, assume a binary classification with labels -1 and 1)
    y_example = 1.0
    
    # FGSM attack demonstration.
    epsilon = 0.1  # Perturbation magnitude for FGSM.
    x_adv_fgsm = fgsm_attack(weights_example, x_example, y_example, epsilon)
    print("Original x:", x_example)
    print("Adversarial x (FGSM):", x_adv_fgsm)
    
    # BIM attack demonstration.
    iterations = 5  # Number of iterative steps.
    alpha = 0.02   # Step size per iteration.
    x_adv_bim = bim_attack(weights_example, x_example, y_example, epsilon, iterations, alpha)
    print("Adversarial x (BIM):", x_adv_bim)
