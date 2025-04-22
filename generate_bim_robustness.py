# File: generate_bim_robustness.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     Evaluate BIM robustness across epsilon values [0.0, 1.0] in 0.05 steps
#     for 3 quantum classifiers: Angle, Amplitude, and Reupload.
#     Save:
#       - Accuracy vs Epsilon plot (combined)
#       - Loss vs Epsilon plot (combined)
#       - Individual .csv files for all metrics
#
# Output Path: C:\Users\ASDF\Desktop\part-2_fyp\Results\bim

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root directory to sys.path
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent)

# Define output path
OUT_DIR = r"C:\Users\ASDF\Desktop\part-2_fyp\Results\bim"
os.makedirs(OUT_DIR, exist_ok=True)

# Epsilon range for BIM attack
epsilons = np.round(np.arange(0.0, 1.0001, 0.05), 2)
bim_iterations = 5
bim_alpha = 0.02

# Import models + adversaries (replace dummy imports with actual ones)
from models.quantum_model_angle import quantum_classifier as qc_angle
from adversaries.adversaries_angle import bim_attack as bim_angle
from models.quantum_model_amplitude import quantum_classifier_amplitude as qc_amp
from adversaries.adversaries_amplitude import bim_attack_amplitude as bim_amp
from models.quantum_model_reupload import quantum_classifier_reupload as qc_reu
from adversaries.adversaries_reupload import bim_attack_reupload as bim_reu

# Config for all models
MODELS = {
    "Angle": {
        "qc": qc_angle,
        "bim": bim_angle,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\angle\angle_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_test.csv"
    },
    "Amplitude": {
        "qc": qc_amp,
        "bim": bim_amp,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\amplitude\amplitude_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_test.csv"
    },
    "Reupload": {
        "qc": qc_reu,
        "bim": bim_reu,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\Reupload\reupload_10layers_cleam.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload\banknote_reupload_preprocessed_test.csv"
    }
}

# Helper to load test set
def load_test(path):
    df = pd.read_csv(path)
    X = df[["variance","skewness","curtosis","entropy"]].values
    y = df["class"].apply(lambda v: -1.0 if v==0 else 1.0).values
    return X, y

# Storage for combined plots
combined_acc = {}
combined_loss = {}

# Style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True
})

# Run BIM evaluation for each model
for name, cfg in MODELS.items():
    qc = cfg["qc"]
    bim = cfg["bim"]
    weights = np.load(cfg["weights"], allow_pickle=True)
    X_test, y_test = load_test(cfg["test_csv"])

    acc_eps, loss_eps = [], []

    for eps in epsilons:
        print(f"[{name}] BIM ε = {eps:.2f}")
        correct, losses = 0, []

        for x, y in zip(X_test, y_test):
            x_adv = bim(weights, x.copy(), y, eps, bim_iterations, bim_alpha)
            pred = qc(weights, x_adv)
            label = 1.0 if pred >= 0 else -1.0
            if label == y:
                correct += 1
            losses.append((pred - y)**2)

        acc_eps.append(correct / len(y_test))
        loss_eps.append(np.mean(losses))

    combined_acc[name] = acc_eps
    combined_loss[name] = loss_eps

    # Save CSV
    df = pd.DataFrame({
        "epsilon": epsilons,
        f"{name.lower()}_accuracy": acc_eps,
        f"{name.lower()}_loss": loss_eps
    })
    df.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_bim_metrics.csv"), index=False)

# Save combined CSV
combined_df = pd.DataFrame({"epsilon": epsilons})
for name in MODELS:
    combined_df[f"{name.lower()}_accuracy"] = combined_acc[name]
    combined_df[f"{name.lower()}_loss"] = combined_loss[name]

combined_df.to_csv(os.path.join(OUT_DIR, "combined_bim_metrics.csv"), index=False)

# Plot Accuracy
plt.figure()
for name in MODELS:
    plt.plot(epsilons, combined_acc[name], marker='o', label=name)
plt.title("BIM Accuracy vs Epsilon (All Embeddings)")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Adversarial Test Accuracy")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "combined_accuracy_vs_eps.png"))
plt.close()

# Plot Loss
plt.figure()
for name in MODELS:
    plt.plot(epsilons, combined_loss[name], marker='o', label=name)
plt.title("BIM Loss vs Epsilon (All Embeddings)")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Adversarial Test Loss (MSE)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "combined_loss_vs_eps.png"))
plt.close()

print(f"[✓] All BIM results saved in {OUT_DIR}")
