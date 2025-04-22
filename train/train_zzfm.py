# File: train_quantum_model_zzfm.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

# Purpose:
#     This script trains a variational quantum classifier that uses the
#     ZZ‐feature map embedding on the Banknote Authentication dataset.
#     It supports three training modes:
#       - clean: no adversarial perturbations,
#       - fgsm: Fast Gradient Sign Method adversarial training,
#       - bim: Basic Iterative Method adversarial training.

# How it works:
#     1. **Configuration & Imports**  
#        - Sets training mode, adversarial hyperparameters (ε, BIM steps/α), 
#          training hyperparameters (epochs, learning rate).
#        - Imports the quantum model (`quantum_classifier_zzfm`) and 
#          adversarial functions (`fgsm_attack_zzfm`, `bim_attack_zzfm`).

#     2. **Data Loading**  
#        - Reads preprocessed CSVs (`train`, `validation`, `test`) from 
#          the `data/ZZ-fm` folder.
#        - Extracts feature vectors and maps class labels {0,1} → {–1,+1}.

#     3. **Loss & Accuracy Definitions**  
#        - Binary cross‑entropy loss on ⟨Z⟩ mapped to [0,1].
#        - Accuracy = fraction of samples for which sign(f(Θ,x)) == y.
#        - Logging raw ⟨Z⟩ distributions for train/val/test each epoch.
#        - Logging gradient‑norm each epoch to detect barren plateaus.

#     4. **Optimizer & Weights Initialization**  
#        - Uses a full‐batch Adam optimizer.
#        - Initializes weights of shape (p, n_qubits, 3) to random small values.

#     5. **Training Loop**  
#        For each epoch:
#          a. **Train step**:  
#             - For each training sample, optionally apply FGSM or BIM 
#               to perturb x.
#             - Compute squared‐error loss and average over all samples.
#             - Update weights via `opt.step_and_cost`.
#          b. **Metrics Computation**:  
#             - Compute loss & accuracy on train, validation, and clean test sets.
#             - Log raw ⟨Z⟩ distributions on train/val/test.
#             - Log gradient‑norm of the full‐batch loss.
#             - If adversarial training, compute loss & accuracy on 
#               adversarially perturbed test set.
#          c. **Logging**:  
#             - Print epoch number, duration, and all metrics.
#          d. **History Recording**:  
#             - Append each metric to a history dictionary.

#     6. **Saving Results**  
#        - Saves each metric history as a CSV in 
#          `Gen_data/ZZfm/clean/` or `Gen_data/ZZfm/adversarial/`.
#        - Saves the final weight array as 
#          `weights/ZZfm/zzfm_<p>layers_<mode>_<ε>.npy`.

# Inputs:
#     - Preprocessed data CSVs in `data/ZZ-fm`:
#         * banknote_zzfm_preprocessed_train.csv
#         * banknote_zzfm_preprocessed_validation.csv
#         * banknote_zzfm_preprocessed_test.csv

# Outputs:
#     - Metric CSVs in `Gen_data/ZZfm/{clean,adversarial}/`
#       (train_loss.csv, train_acc.csv, val_loss.csv, …, z_dist_train.csv, grad_norm.csv).
#     - Final weights in `weights/ZZfm/`.

# Notes:
#     - The script uses PennyLane for quantum circuit evaluation.
#     - Make sure `quantum_model_zzfm.py` and `adversaries_zzfm.py` are in PYTHONPATH.

import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import time
import logging
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Import model and adversarial attack functions
from models.quantum_model_zzfm import quantum_classifier_zzfm, n_qubits, p as num_layers
from adversaries.adversaries_zzfm import fgsm_attack_zzfm, bim_attack_zzfm

# ------------------------------
# 1. HYPERPARAMETERS
# ------------------------------
training_type  = "clean"    # Options: "clean", "fgsm", "bim"
epsilon        = 0.1        # Perturbation magnitude for FGSM/BIM
bim_iterations = 5          # BIM steps
bim_alpha      = 0.02       # BIM per‐step size
num_epochs     = 50         # Training epochs
learning_rate  = 0.01       # Adam learning rate

# ------------------------------
# 2. PATHS
# ------------------------------
data_dir    = r"C:\Users\ASDF\Desktop\part-2_fyp\data\ZZ-fm"
train_csv   = os.path.join(data_dir, "banknote_zzfm_preprocessed_train.csv")
val_csv     = os.path.join(data_dir, "banknote_zzfm_preprocessed_validation.csv")
test_csv    = os.path.join(data_dir, "banknote_zzfm_preprocessed_test.csv")

metrics_base = r"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\ZZfm"
metrics_dir  = os.path.join(metrics_base, "clean" if training_type=="clean" else "adversarial")
os.makedirs(metrics_dir, exist_ok=True)

weights_dir  = r"C:\Users\ASDF\Desktop\part-2_fyp\weights\ZZfm"
os.makedirs(weights_dir, exist_ok=True)

logging.info("Mode: %s", training_type)
logging.info("Metrics → %s", metrics_dir)
logging.info("Weights → %s", weights_dir)

# ------------------------------
# 3. DATA LOADING
# ------------------------------
def load_split(path):
    df = pd.read_csv(path)
    X  = df[["variance","skewness","curtosis","entropy"]].values
    y  = df["class"].apply(lambda v: -1.0 if v==0 else 1.0).values
    return X, y

X_train, y_train = load_split(train_csv)
X_val,   y_val   = load_split(val_csv)
X_test,  y_test  = load_split(test_csv)
logging.info("Loaded data: train=%d, val=%d, test=%d",
             len(y_train), len(y_val), len(y_test))

# ------------------------------
# 4. LOSS & ACCURACY
# ------------------------------
def cost(weights, X, y):
    losses = []
    for xi, yi in zip(X, y):
        # get raw model output in [–1,1]
        z = quantum_classifier_zzfm(weights, xi)
        # map to probability p = (z+1)/2 in [0,1]
        p = (z + 1.0) / 2.0
        # clamp to avoid log(0)
        p = pnp.clip(p, 1e-6, 1-1e-6)
        # binary cross‐entropy:  yi==+1 → target=1, yi==–1 → target=0
        target = (yi + 1.0) / 2.0
        losses.append(
            - ( target * pnp.log(p) + (1-target) * pnp.log(1-p) )
        )
    return pnp.mean(pnp.stack(losses))

def accuracy(weights, X, y):
    correct = sum((1.0 if quantum_classifier_zzfm(weights, xi)>=0 else -1.0)==yi
                  for xi, yi in zip(X, y))
    return correct / len(y)

# ------------------------------
# 5. OPTIMIZER & WEIGHTS
# ------------------------------
# Initialize near‐identity: angles in [–π/8, +π/8]
init_scale = np.pi / 8
weights = pnp.array(
    np.random.uniform(-init_scale, init_scale, (num_layers, n_qubits, 3)),
    requires_grad=True
)

opt     = qml.AdamOptimizer(stepsize=learning_rate)
logging.info("Initialized weights of shape %s", weights.shape)

# Pre‐allocate files for raw ⟨Z⟩ and gradient norms
z_train_log = os.path.join(metrics_dir, "z_dist_train.csv")
z_val_log   = os.path.join(metrics_dir, "z_dist_val.csv")
z_test_log  = os.path.join(metrics_dir, "z_dist_test.csv")
grad_log    = os.path.join(metrics_dir, "grad_norm.csv")

# ------------------------------
# 6. TRAINING LOOP
# ------------------------------
history = {key: [] for key in (
    "train_loss","train_acc","val_loss","val_acc","test_loss","test_acc",
    "adv_test_loss","adv_test_acc"
)}
z_train_hist, z_val_hist, z_test_hist, grad_norm_hist = [], [], [], []

for epoch in range(1, num_epochs+1):
    start = time.time()

    # a) Training step
    def step_fn(w):
        losses = []
        for xi, yi in zip(X_train, y_train):
            x_in = xi.copy()
            if training_type=="fgsm":
                x_in = fgsm_attack_zzfm(w, x_in, yi, epsilon)
            elif training_type=="bim":
                x_in = bim_attack_zzfm(w, x_in, yi,
                                     epsilon, bim_iterations, bim_alpha)
            losses.append((quantum_classifier_zzfm(w, x_in) - yi)**2)
        return pnp.mean(pnp.array(losses))

    weights, _ = opt.step_and_cost(step_fn, weights)

    # b) Compute metrics
    tr_l = cost(weights, X_train, y_train); tr_a = accuracy(weights, X_train, y_train)
    v_l  = cost(weights, X_val,   y_val);   v_a  = accuracy(weights, X_val,   y_val)
    te_l = cost(weights, X_test,  y_test);  te_a = accuracy(weights, X_test,  y_test)

    # ─── Raw ⟨Z⟩ distributions ──────────────────────────────────────────────
    z_train = [quantum_classifier_zzfm(weights, x) for x in X_train]
    z_val   = [quantum_classifier_zzfm(weights, x) for x in X_val]
    z_test  = [quantum_classifier_zzfm(weights, x) for x in X_test]
    z_train_hist.append((np.mean(z_train), np.std(z_train)))
    z_val_hist.append((np.mean(z_val),   np.std(z_val)))
    z_test_hist.append((np.mean(z_test), np.std(z_test)))
    logging.info("⟨Z⟩ train: mean=%.3f std=%.3f | val: mean=%.3f std=%.3f | test: mean=%.3f std=%.3f",
                 *z_train_hist[-1], *z_val_hist[-1], *z_test_hist[-1])
    # ─────────────────────────────────────────────────────────────────────────

    # ─── Gradient‐norm logging ───────────────────────────────────────────────
    grad_fn = qml.grad(cost, argnum=0)
    grads = grad_fn(weights, X_train, y_train)
    gn = float(pnp.linalg.norm(grads))
    grad_norm_hist.append(gn)
    logging.info("Gradient norm: %.3e", gn)
    # ─────────────────────────────────────────────────────────────────────────

    # c) Adversarial test metrics
    if training_type!="clean":
        adv_losses, adv_corr = [], 0
        for xi, yi in zip(X_test, y_test):
            xa = fgsm_attack_zzfm(weights, xi.copy(), yi, epsilon) if training_type=="fgsm" \
               else bim_attack_zzfm(weights, xi.copy(), yi,
                                   epsilon, bim_iterations, bim_alpha)
            adv_losses.append((quantum_classifier_zzfm(weights, xa) - yi)**2)
            if (1.0 if quantum_classifier_zzfm(weights, xa)>=0 else -1.0)==yi:
                adv_corr += 1
        adv_l = pnp.mean(pnp.array(adv_losses))
        adv_a = adv_corr / len(y_test)
    else:
        adv_l, adv_a = None, None

    # d) Record
    for key, val in zip(history.keys(), 
                        (tr_l, tr_a, v_l, v_a, te_l, te_a, adv_l, adv_a)):
        if val is not None:
            history[key].append(val)

    # e) Log epoch summary
    logging.info(
        "Epoch %3d/%d (%.1fs)  tr=[%.3f,%.1f%%]  val=[%.3f,%.1f%%]  test=[%.3f,%.1f%%]  adv=[%s,%s%%]",
        epoch, num_epochs, time.time()-start,
        tr_l, tr_a*100, v_l, v_a*100, te_l, te_a*100,
        f"{adv_l:.3f}" if adv_l is not None else "--",
        f"{adv_a*100:.1f}" if adv_a is not None else "--"
    )

# ------------------------------
# 7. SAVE METRICS
# ------------------------------
# Standard metrics
for metric, vals in history.items():
    if "adv_" in metric and training_type=="clean":
        continue
    pd.DataFrame({metric: vals}).to_csv(
        os.path.join(metrics_dir, f"{metric}.csv"), index=False
    )
# Raw ⟨Z⟩ and gradient norms
pd.DataFrame(z_train_hist, columns=["z_train_mean","z_train_std"]).to_csv(z_train_log, index=False)
pd.DataFrame(z_val_hist,   columns=["z_val_mean",  "z_val_std"]  ).to_csv(z_val_log,   index=False)
pd.DataFrame(z_test_hist,  columns=["z_test_mean", "z_test_std"] ).to_csv(z_test_log,  index=False)
pd.DataFrame({"grad_norm": grad_norm_hist}).to_csv(grad_log, index=False)

logging.info("Metrics saved to %s", metrics_dir)

# ------------------------------
# 8. SAVE WEIGHTS
# ------------------------------
suffix = "clean" if training_type=="clean" else f"{training_type}_{epsilon}"
filename = f"zzfm_{num_layers}layers_{suffix}.npy"
pnp.save(os.path.join(weights_dir, filename), weights)
logging.info("Weights saved to %s", os.path.join(weights_dir, filename))
