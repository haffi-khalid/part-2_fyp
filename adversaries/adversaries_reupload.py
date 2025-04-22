# File: adversaries_reupload.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]
#
# Purpose:
#     Implements white‑box adversarial attacks (FGSM and BIM) for the
#     data re‑uploading quantum classifier defined in quantum_model_reupload.py.
#     - fgsm_attack_reupload: one‑step Fast Gradient Sign Method.
#     - bim_attack_reupload: iterative Basic Iterative Method.
#
# Requirements:
#     - PennyLane for autodifferentiation.
#     - quantum_model_reupload.py on PYTHONPATH.

import pennylane as qml
from pennylane import numpy as np
from models.quantum_model_reupload import quantum_classifier_reupload  # Ensure module path is correct

# Valid input range for RY gates
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

def loss_reupload(weights, x, y):
    """
    Binary cross‑entropy loss for re‑uploading classifier.

    Args:
        weights (array): variational parameters shape (p, 5, 3)
        x       (array): feature vector length‑4, values in [0,π]
        y       (float):  label in {-1,+1}

    Returns:
        float: cross‑entropy loss.
    """
    # raw expectation ∈ [–1,1]
    z = quantum_classifier_reupload(weights, x)
    # map to probability p∈(0,1)
    p = (z + 1.0) / 2.0
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # target t∈{0,1}
    t = (y + 1.0) / 2.0
    return - (t * np.log(p) + (1 - t) * np.log(1 - p))

def fgsm_attack_reupload(weights, x, y, epsilon):
    """
    Generate an adversarial example via FGSM.

    x_adv = clip(x + ε * sign(∇_x L), 0, π)

    Args:
        weights (array): current variational parameters
        x       (array): original input
        y       (float): true label
        epsilon (float): perturbation magnitude

    Returns:
        array: adversarial example in [0,π]^4
    """
    grad_fn = qml.grad(loss_reupload, argnum=1)
    grad_x = grad_fn(weights, x, y)
    x_adv = x + epsilon * np.sign(grad_x)
    return np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)

def bim_attack_reupload(weights, x, y, epsilon, iterations, alpha):
    """
    Generate an adversarial example via BIM (iterative FGSM).

    Repeatedly apply FGSM steps of size α, clipping total perturbation to ε.

    Args:
        weights    (array): variational parameters
        x          (array): original input
        y          (float): true label
        epsilon    (float): max total perturbation
        iterations (int): number of FGSM steps
        alpha      (float): step size per iteration

    Returns:
        array: adversarial example in [0,π]^4
    """
    x_adv = x.copy()
    for i in range(iterations):
        grad_fn = qml.grad(loss_reupload, argnum=1)
        grad_x = grad_fn(weights, x_adv, y)
        x_adv = x_adv + alpha * np.sign(grad_x)
        # clip to ensure ∥x_adv - x∥∞ ≤ epsilon
        perturb = np.clip(x_adv - x, -epsilon, epsilon)
        x_adv = x + perturb
        # clip to valid [0,π]
        x_adv = np.clip(x_adv, ANGLE_MIN, ANGLE_MAX)
    return x_adv

# Example usage (can be removed later)
if __name__ == "__main__":
    import numpy as onp
    # dummy example
    x_ex = onp.array([0.2*onp.pi,0.4*onp.pi,0.6*onp.pi,0.8*onp.pi])
    from models.quantum_model_reupload import p as depth, n_qubits
    w_ex = np.random.uniform(0, 2*onp.pi, (depth, n_qubits, 3))
    y_ex = 1.0
    x_fgsm = fgsm_attack_reupload(w_ex, x_ex, y_ex, 0.1)
    x_bim  = bim_attack_reupload(w_ex, x_ex, y_ex, 0.1, 5, 0.02)
    print("FGSM adversarial:", x_fgsm)
    print("BIM adversarial: ", x_bim)
