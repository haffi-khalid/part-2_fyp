"""
File: quantum_adversarial_attacks_amplitude.py
Author: Muhammad Haffi Khalid
Date: [Today's Date]

Purpose:
    Implements adversarial attack methods (FGSM and BIM) for the amplitude‐encoded
    quantum classifier. These functions can be used within the training loop to
    generate adversarial examples for both training and testing in adversarial training.

Functions:
    - cost_function_amplitude(weights, x, y)
    - fgsm_attack_amplitude(weights, x, y, epsilon)
    - bim_attack_amplitude(weights, x, y, epsilon, iterations, alpha)

Usage:
    from quantum_adversarial_attacks_amplitude import fgsm_attack_amplitude, bim_attack_amplitude
    x_adv = fgsm_attack_amplitude(weights, x_clean, y_true, epsilon)
"""

import logging
import pennylane as qml
from pennylane import numpy as np

# Import the amplitude model classifier
try:
    from models.quantum_model_amplitude import quantum_classifier_amplitude
    logging.info("Imported quantum_classifier_amplitude successfully.")
except ImportError as e:
    logging.error("Failed to import quantum_classifier_amplitude: %s", e)
    raise

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

def cost_function_amplitude(weights, x, y):
    """
    Squared error loss for the amplitude‐encoded quantum classifier:
        L = (quantum_classifier_amplitude(weights, x) - y)^2

    Args:
        weights (array): Variational parameters of shape (p, 3, 3).
        x       (array): Input feature vector (length 4, unit‐normalized).
        y       (float): True label (-1 or +1).

    Returns:
        float: Loss value.
    """
    try:
        pred = quantum_classifier_amplitude(weights, x)
        return (pred - y) ** 2
    except Exception as e:
        logging.error("Error in cost_function_amplitude: %s", e)
        raise

def fgsm_attack_amplitude(weights, x, y, epsilon):
    """
    Fast Gradient Sign Method (FGSM) for amplitude‐encoded classifier.

    x_adv = clip( x + epsilon * sign(∇_x L), -amplitude_bound, +amplitude_bound )

    Since amplitude encoding requires x to remain unit‐norm, after FGSM we
    re‐normalize x_adv to unit norm.

    Args:
        weights (array): Model parameters.
        x       (array): Original input (||x||₂ = 1).
        y       (float): True label.
        epsilon (float): Perturbation magnitude (small).

    Returns:
        array: Adversarial input x_adv (unit‐normalized).
    """
    try:
        # Gradient of loss w.r.t. input x
        grad_fn = qml.grad(cost_function_amplitude, argnum=1)
        grad_x = grad_fn(weights, x, y)
        # FGSM step
        x_adv = x + epsilon * np.sign(grad_x)
        # Renormalize to unit norm
        norm = np.linalg.norm(x_adv)
        if norm == 0:
            logging.warning("Zero vector encountered in FGSM; returning original x")
            return x
        return x_adv / norm
    except Exception as e:
        logging.error("Error in fgsm_attack_amplitude: %s", e)
        raise

def bim_attack_amplitude(weights, x, y, epsilon, iterations, alpha):
    """
    Basic Iterative Method (BIM) for amplitude‐encoded classifier.

    Iteratively applies FGSM with step size alpha, clipping the total perturbation
    to ||x_adv - x||_∞ ≤ epsilon, and re‐normalizing at each step to maintain unit norm.

    Args:
        weights    (array): Model parameters.
        x          (array): Original input (||x||₂ = 1).
        y          (float): True label.
        epsilon    (float): Maximum total perturbation (∞-norm).
        iterations (int):   Number of BIM steps.
        alpha      (float): Step size per iteration.

    Returns:
        array: Adversarial input x_adv (unit‐normalized).
    """
    x_adv = x.copy()
    try:
        for i in range(iterations):
            grad_fn = qml.grad(cost_function_amplitude, argnum=1)
            grad_x = grad_fn(weights, x_adv, y)
            # BIM step
            x_adv = x_adv + alpha * np.sign(grad_x)
            # Clip ∞-norm perturbation
            perturb = x_adv - x
            perturb = np.clip(perturb, -epsilon, epsilon)
            x_adv = x + perturb
            # Renormalize
            norm = np.linalg.norm(x_adv)
            if norm == 0:
                logging.warning("Zero vector at BIM iteration %d; using original x", i)
                x_adv = x.copy()
                break
            x_adv = x_adv / norm
        return x_adv
    except Exception as e:
        logging.error("Error in bim_attack_amplitude (iteration %d): %s", i, e)
        raise
