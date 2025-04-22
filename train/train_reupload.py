# File: train_reupload.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]
#
# Purpose:
#     Train the data re‑uploading variational quantum classifier on the
#     Banknote Authentication dataset with options for:
#       - clean training,
#       - adversarial training via FGSM,
#       - adversarial training via BIM.
#
#     Logs per‑epoch metrics: loss, accuracy, ⟨Z⟩ distributions, gradient norms,
#     and (if adversarial) adversarial test metrics.  Saves metrics as CSVs
#     and final weights for later analysis.
#
# Usage:
#     python train_reupload.py
#
# Outputs:
#     - Metrics CSVs in Gen_data/Reupload/{clean,adversarial}/
#     - Final weights in weights/Reupload/

import os
import sys
import time
import logging

import numpy as onp
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

# Add project root to PYTHONPATH
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent)

from models.quantum_model_reupload import quantum_classifier_reupload, n_qubits, p as num_layers
from adversaries.adversaries_reupload import fgsm_attack_reupload, bim_attack_reupload

# ------------------------------
# 1. HYPERPARAMETERS
# ------------------------------
training_type  = "bim"      # Options: "clean", "fgsm", "bim"
epsilon        = 0.1          # FGSM/BIM perturbation magnitude
bim_iterations = 3            # BIM steps
bim_alpha      = 0.02         # BIM per‑step size
num_epochs     = 100          # Total epochs
learning_rate  = 0.01         # Adam step size

# ------------------------------
# 2. PATHS
# ------------------------------
data_dir    = r"C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload"
train_csv   = os.path.join(data_dir, "banknote_reupload_preprocessed_train.csv")
val_csv     = os.path.join(data_dir, "banknote_reupload_preprocessed_validation.csv")
test_csv    = os.path.join(data_dir, "banknote_reupload_preprocessed_test.csv")

metrics_base = r"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\Reupload"
metrics_dir  = os.path.join(metrics_base, "clean" if training_type=="clean" else "adversarial")
os.makedirs(metrics_dir, exist_ok=True)

weights_dir  = r"C:\Users\ASDF\Desktop\part-2_fyp\weights\Reupload"
os.makedirs(weights_dir, exist_ok=True)

# ------------------------------
# 3. LOGGING CONFIGURATION
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
logging.info("Training mode: %s", training_type)
logging.info("Metrics directory: %s", metrics_dir)
logging.info("Weights directory: %s", weights_dir)

# ------------------------------
# 4. DATA LOADING
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
# 5. LOSS & ACCURACY
# ------------------------------
def cost(weights, X, y):
    losses = []
    for xi, yi in zip(X, y):
        # raw ⟨Z⟩ ∈ [–1,1]
        z = quantum_classifier_reupload(weights, xi)
        # map to p∈(0,1)
        p = (z + 1.0) / 2.0
        p = pnp.clip(p, 1e-6, 1-1e-6)
        t = (yi + 1.0) / 2.0
        losses.append(- (t * pnp.log(p) + (1-t) * pnp.log(1-p)))
    return pnp.mean(pnp.stack(losses))

def accuracy(weights, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        pred = quantum_classifier_reupload(weights, xi)
        label = 1.0 if pred >= 0 else -1.0
        if label == yi:
            correct += 1
    return correct / len(y)

# ------------------------------
# 6. OPTIMIZER & WEIGHTS INIT
# ------------------------------
init_scale = onp.pi / 8
weights = pnp.array(onp.random.uniform(-init_scale, init_scale,
                                       (num_layers, n_qubits, 3)),
                    requires_grad=True)
opt = qml.AdamOptimizer(stepsize=learning_rate)
logging.info("Initialized weights with shape %s", weights.shape)

# Prepare logs for raw ⟨Z⟩ and gradient norm
z_train_hist, z_val_hist, z_test_hist, grad_norm_hist = [], [], [], []

# ------------------------------
# 7. TRAINING LOOP
# ------------------------------
history = {k: [] for k in ("train_loss","train_acc","val_loss","val_acc",
                           "test_loss","test_acc","adv_test_loss","adv_test_acc")}

for epoch in range(1, num_epochs+1):
    t0 = time.time()

    # a) Training step (full‐batch)
    def step_fn(w):
        batch_losses = []
        for xi, yi in zip(X_train, y_train):
            x_in = xi.copy()
            if training_type == "fgsm":
                x_in = fgsm_attack_reupload(w, x_in, yi, epsilon)
            elif training_type == "bim":
                x_in = bim_attack_reupload(w, x_in, yi,
                                           epsilon, bim_iterations, bim_alpha)
            batch_losses.append((quantum_classifier_reupload(w, x_in) - yi)**2)
        return pnp.mean(pnp.array(batch_losses))

    weights, _ = opt.step_and_cost(step_fn, weights)

    # b) Compute clean metrics
    tr_l = cost(weights, X_train, y_train); tr_a = accuracy(weights, X_train, y_train)
    v_l  = cost(weights, X_val,   y_val);   v_a  = accuracy(weights, X_val,   y_val)
    te_l = cost(weights, X_test,  y_test);  te_a = accuracy(weights, X_test,  y_test)

    # c) Raw ⟨Z⟩ distributions
    z_tr = [quantum_classifier_reupload(weights, x) for x in X_train]
    z_v  = [quantum_classifier_reupload(weights, x) for x in X_val]
    z_te = [quantum_classifier_reupload(weights, x) for x in X_test]
    z_train_hist.append((onp.mean(z_tr), onp.std(z_tr)))
    z_val_hist.append((onp.mean(z_v),   onp.std(z_v)))
    z_test_hist.append((onp.mean(z_te),  onp.std(z_te)))
    logging.info("⟨Z⟩ train: mean=%.3f std=%.3f | val: %.3f,%.3f | test: %.3f,%.3f",
                 *z_train_hist[-1], *z_val_hist[-1], *z_test_hist[-1])

    # d) Gradient norm
    grad_fn = qml.grad(cost, argnum=0)
    grads = grad_fn(weights, X_train, y_train)
    gn = float(pnp.linalg.norm(grads))
    grad_norm_hist.append(gn)
    logging.info("Gradient norm: %.3e", gn)

    # e) Adversarial test metrics
    if training_type != "clean":
        adv_losses, adv_corr = [], 0
        for xi, yi in zip(X_test, y_test):
            xa = (fgsm_attack_reupload(weights, xi.copy(), yi, epsilon)
                  if training_type=="fgsm"
                  else bim_attack_reupload(weights, xi.copy(), yi,
                                           epsilon, bim_iterations, bim_alpha))
            adv_losses.append((quantum_classifier_reupload(weights, xa) - yi)**2)
            if (quantum_classifier_reupload(weights, xa) >= 0 and yi==1) or \
               (quantum_classifier_reupload(weights, xa) <  0 and yi==-1):
                adv_corr += 1
        adv_l = pnp.mean(pnp.array(adv_losses))
        adv_a = adv_corr / len(y_test)
    else:
        adv_l, adv_a = None, None

    # f) Record history
    for key, val in zip(history.keys(),
                        (tr_l, tr_a, v_l, v_a, te_l, te_a, adv_l, adv_a)):
        if val is not None:
            history[key].append(val)

    # g) Log epoch summary
    logging.info(
        "Epoch %2d/%d (%.1fs)  tr=[%.3f,%.1f%%]  val=[%.3f,%.1f%%]  test=[%.3f,%.1f%%]  adv=[%s,%s%%]",
        epoch, num_epochs, time.time()-t0,
        tr_l, tr_a*100, v_l, v_a*100, te_l, te_a*100,
        f"{adv_l:.3f}" if adv_l is not None else "--",
        f"{adv_a*100:.1f}" if adv_a is not None else "--"
    )

# ------------------------------
# 8. SAVE METRICS
# ------------------------------
# per‑epoch metrics
for metric, vals in history.items():
    if metric.startswith("adv_") and training_type=="clean":
        continue
    pd.DataFrame({metric: vals}).to_csv(os.path.join(metrics_dir, f"{metric}.csv"), index=False)

# raw ⟨Z⟩ & gradient norm
pd.DataFrame(z_train_hist, columns=["z_train_mean","z_train_std"]).to_csv(
    os.path.join(metrics_dir, "z_train_dist.csv"), index=False)
pd.DataFrame(z_val_hist, columns=["z_val_mean","z_val_std"]).to_csv(
    os.path.join(metrics_dir, "z_val_dist.csv"), index=False)
pd.DataFrame(z_test_hist, columns=["z_test_mean","z_test_std"]).to_csv(
    os.path.join(metrics_dir, "z_test_dist.csv"), index=False)
pd.DataFrame({"grad_norm": grad_norm_hist}).to_csv(
    os.path.join(metrics_dir, "grad_norm.csv"), index=False)

logging.info("Metrics saved to %s", metrics_dir)

# ------------------------------
# 9. SAVE FINAL WEIGHTS
# ------------------------------
suffix = training_type if training_type=="clean" else f"{training_type}_{epsilon}"
filename = f"reupload_{num_layers}layers_{suffix}.npy"
pnp.save(os.path.join(weights_dir, filename), weights)
logging.info("Weights saved to %s", os.path.join(weights_dir, filename))
