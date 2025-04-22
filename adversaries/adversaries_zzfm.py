# File: adversaries_zzfm.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Adversarial Attacks for ZZ-Feature-Map Quantum Classifier

# Provides:
#   - cost_function_zzfm
#   - fgsm_attack_zzfm
#   - bim_attack_zzfm

# These generate adversarial examples for training and evaluation.
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import logging
import pennylane as qml
from pennylane import numpy as np

# Import the ZZFM classifier
from models.quantum_model_zzfm import quantum_classifier_zzfm

# Valid angle range for ZZFM embedding
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

def cost_function_zzfm(weights, x, y):
    """Squared error loss for ZZFM classifier."""
    try:
        pred = quantum_classifier_zzfm(weights, x)
        return (pred - y) ** 2
    except Exception as e:
        logging.error("Cost function error: %s", e)
        raise

def fgsm_attack_zzfm(weights, x, y, epsilon):
    """
    FGSM for ZZFM: one‐step sign‐gradient perturbation.
    Clips final angles to [0, π].
    """
    try:
        grad_fn = qml.grad(cost_function_zzfm, argnum=1)
        grad_x = grad_fn(weights, x, y)
        x_adv = x + epsilon * np.sign(grad_x)
        # Clip into [0, π]
        return np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)
    except Exception as e:
        logging.error("FGSM attack error: %s", e)
        raise

def bim_attack_zzfm(weights, x, y, epsilon, iterations, alpha):
    """
    BIM for ZZFM: iterative FGSM, with ∞‐norm clipping to ε
    and angle clipping to [0, π] each step.
    """
    x_adv = x.copy()
    try:
        for i in range(iterations):
            grad_fn = qml.grad(cost_function_zzfm, argnum=1)
            grad_x = grad_fn(weights, x_adv, y)
            # step
            x_adv = x_adv + alpha * np.sign(grad_x)
            # ∞-norm clip
            delta = np.clip(x_adv - x, -epsilon, epsilon)
            x_adv = x + delta
            # angle clip
            x_adv = np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)
        return x_adv
    except Exception as e:
        logging.error("BIM attack iteration %d error: %s", i, e)
        raise
