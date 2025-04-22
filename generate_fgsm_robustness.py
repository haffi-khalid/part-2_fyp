# File: generate_fgsm_robustness_subset.py
# Author: Muhammad Haffi Khalid
# Date: April 2025
#
# Purpose:
#     This script evaluates the robustness of three quantum classifiers—
#     Angle, Amplitude, and Reupload embeddings—against FGSM adversarial attacks.
#
#     For ε ∈ [0, 1] (step 0.05), it:
#       - Generates FGSM adversarial test samples.
#       - Computes adversarial test accuracy and MSE loss.
#       - Plots:
#           • Accuracy vs ε  (per model + combined)
#           • Loss vs ε      (per model + combined)
#
#     NOTE: ZZ-Feature Map is excluded in this comparison.





# Ensure project root is on PYTHONPATH
import sys
import os
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Import classifiers and FGSM attack methods
from models.quantum_model_angle import quantum_classifier as qc_angle
from adversaries.adversaries_angle import fgsm_attack as fgsm_angle
from models.quantum_model_amplitude import quantum_classifier_amplitude as qc_amp
from adversaries.adversaries_amplitude import fgsm_attack_amplitude as fgsm_amp
from models.quantum_model_reupload import quantum_classifier_reupload as qc_reu
from adversaries.adversaries_reupload import fgsm_attack_reupload as fgsm_reu

# Model configs
MODELS = {
    "Angle": {
        "qc": qc_angle,
        "fgsm": fgsm_angle,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\angle\angle_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_test.csv"
    },
    "Amplitude": {
        "qc": qc_amp,
        "fgsm": fgsm_amp,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\amplitude\amplitude_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude\banknote_amplitude_preprocessed_test.csv"
    },
    "Reupload": {
        "qc": qc_reu,
        "fgsm": fgsm_reu,
        "weights": r"C:\Users\ASDF\Desktop\part-2_fyp\weights\Reupload\reupload_10layers_clean.npy",
        "test_csv": r"C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload\banknote_reupload_preprocessed_test.csv"
    }
}

# Output directory for all plots
OUT_DIR = r"C:\Users\ASDF\Desktop\part-2_fyp\Results\fgsm"
os.makedirs(OUT_DIR, exist_ok=True)

# FGSM epsilon range
epsilons = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# Store results for combination plots
combined_acc = {}
combined_loss = {}

# Load test data
def load_test_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    y = df["class"].apply(lambda v: -1.0 if v == 0 else 1.0).values
    return X, y

# Plot style settings
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Loop through models
for name, config in MODELS.items():
    qc_func = config["qc"]
    fgsm_func = config["fgsm"]
    weights = np.load(config["weights"], allow_pickle=True)
    X_test, y_test = load_test_data(config["test_csv"])

    acc_eps = []
    loss_eps = []

    for eps in epsilons:
        print(f"[{name}] Evaluating FGSM at epsilon = {eps:.2f}...")
        correct = 0
        mse_vals = []

        for x, y in zip(X_test, y_test):
            x_adv = fgsm_func(weights, x.copy(), y, eps)
            pred = qc_func(weights, x_adv)
            label = 1.0 if pred >= 0 else -1.0
            if label == y:
                correct += 1
            mse_vals.append((pred - y) ** 2)

        acc_eps.append(correct / len(y_test))
        loss_eps.append(np.mean(mse_vals))

    combined_acc[name] = acc_eps
    combined_loss[name] = loss_eps

        # Save CSV for this embedding
    df = pd.DataFrame({
        "epsilon": epsilons,
        f"{name.lower()}_accuracy": acc_eps,
        f"{name.lower()}_loss": loss_eps
    })
    df.to_csv(os.path.join(OUT_DIR, f"{name.lower()}_fgsm_metrics.csv"), index=False)
    print(f"[{name}] Metrics saved to → {name.lower()}_fgsm_metrics.csv")


    # Plot Accuracy
    plt.figure()
    plt.plot(epsilons, acc_eps, marker='o')
    plt.title(f"{name} Embedding: FGSM Accuracy vs Epsilon")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Adversarial Test Accuracy (%)")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUT_DIR, f"{name.lower()}_acc_vs_eps.png"))
    plt.close()

    # Plot Loss
    plt.figure()
    plt.plot(epsilons, loss_eps, marker='o')
    plt.title(f"{name} Embedding: FGSM Loss vs Epsilon")
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("MSE Loss")
    plt.savefig(os.path.join(OUT_DIR, f"{name.lower()}_loss_vs_eps.png"))
    plt.close()

# Combined Accuracy Plot
plt.figure()
for name, acc in combined_acc.items():
    plt.plot(epsilons, acc, marker='o', label=name)
plt.title("FGSM Accuracy vs Epsilon (All Embeddings)")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Adversarial Test Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "combined_accuracy_vs_eps.png"))
plt.close()

# Combined Loss Plot
plt.figure()
for name, loss in combined_loss.items():
    plt.plot(epsilons, loss, marker='o', label=name)
plt.title("FGSM Loss vs Epsilon (All Embeddings)")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Adversarial Test Loss (MSE)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "combined_loss_vs_eps.png"))
plt.close()

# Save combined CSV for all
combined_df = pd.DataFrame({"epsilon": epsilons})
for name in MODELS:
    combined_df[f"{name.lower()}_accuracy"] = combined_acc[name]
    combined_df[f"{name.lower()}_loss"] = combined_loss[name]

combined_df.to_csv(os.path.join(OUT_DIR, "combined_fgsm_metrics.csv"), index=False)
print("[✓] Combined CSV saved as combined_fgsm_metrics.csv")
print(f"[✓] All plots and data saved in {OUT_DIR}")
