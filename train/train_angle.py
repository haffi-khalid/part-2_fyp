# # File: train_quantum_model_angle.py
# # Author: Muhammad Haffi Khalid
# # Date: [Today's Date]

# # Purpose:
# #     This training script supports three training modes for a quantum classifier with angle encoding:
# #       1. Clean training (no adversarial perturbations)
# #       2. Adversarial training using FGSM
# #       3. Adversarial training using BIM

# #     For adversarial training, FGSM/BIM perturbations are applied to both training and test data
# #     (using a user-defined epsilon). Additionally, training metrics (training loss, training accuracy,
# #     validation loss, validation accuracy, test loss, test accuracy) and for adversarial training, extra
# #     metrics (adversarial test loss and adversarial test accuracy) are saved as CSV files in distinct folders.
# #     At each epoch, all metrics and the epoch runtime are logged.

# #     Finally, the trained weights are saved in a designated weights folder with a name indicating the type of training.
    
# # Usage:
# #     Ensure that the following files are available:
# #       - Preprocessed CSV files for train, validation, and test (paths specified below)
# #       - "quantum_model_angle.py" (quantum classifier model)
# #       - "quantum_adversarial_attacks.py" (FGSM and BIM functions)
    
# #     Adjust parameters and file paths as needed.

import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import time
import pandas as pd
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Import the quantum classifier and adversarial attack functions.
from models.quantum_model_angle import quantum_classifier, n_qubits, p as num_layers  # p is the number of layers in the model.
from adversaries.adversaries_angle import fgsm_attack, bim_attack

# ------------------------------
# 1. PARAMETERS (Adjust as needed)
# ------------------------------

# Training mode: "clean", "fgsm", or "bim"
training_type = "bim"  # Options: "clean", "fgsm", "bim"

# Adversarial attack hyperparameters (used if training_type is not "clean")
epsilon = 0.1           # Maximum perturbation magnitude
bim_iterations = 3      # Number of iterations for BIM
bim_alpha = 0.01        # Step size per iteration for BIM

# Training hyperparameters
num_epochs = 100         # Total number of training epochs
learning_rate = 0.05    # Learning rate for the Adam optimizer

# ------------------------------
# 2. DATA PATHS
# ------------------------------
train_data_path = r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_train.csv"
val_data_path = r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_validation.csv"
test_data_path = r"C:\Users\ASDF\Desktop\part-2_fyp\data\angle\banknote_angle_preprocessed_test.csv"

# Output folders for metrics
if training_type == "clean":
    metrics_folder = r"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\angle\clean"
else:
    metrics_folder = r"C:\Users\ASDF\Desktop\part-2_fyp\Gen_data\angle\adversarial"

if not os.path.exists(metrics_folder):
    os.makedirs(metrics_folder)

# Weights save folder
weights_folder = r"C:\Users\ASDF\Desktop\part-2_fyp\weights\angle"
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

# ------------------------------
# 3. DATA LOADING AND PREPARATION
# ------------------------------
# Load preprocessed CSV files.
df_train = pd.read_csv(train_data_path)
df_val = pd.read_csv(val_data_path)
df_test = pd.read_csv(test_data_path)

# Extract features and labels.
# The dataset columns are assumed to be: ["variance", "skewness", "curtosis", "entropy", "class"]
# Convert the 'class' column: assume original labels {0,1} and map to {-1, +1} for compatibility with PauliZ measurement.
def process_dataframe(df):
    X = df[["variance", "skewness", "curtosis", "entropy"]].values
    # Normalize class: 0 becomes -1, 1 becomes +1.
    y = df["class"].apply(lambda v: -1.0 if v == 0 else 1.0).values.astype(np.float64)
    return X, y

X_train, y_train = process_dataframe(df_train)
X_val, y_val = process_dataframe(df_val)
X_test, y_test = process_dataframe(df_test)

# ------------------------------
# 4. DEFINE COST FUNCTION AND METRICS
# ------------------------------
# We define the cost function for training (mean squared error).
def cost(weights, X, y):
    losses = []
    for x_val, y_val in zip(X, y):
        loss = (quantum_classifier(weights, x_val) - y_val)**2
        losses.append(loss)
    return pnp.mean(pnp.array(losses))

# Define an accuracy function (for binary classification with threshold 0)
def accuracy(weights, X, y):
    correct = 0
    for x_val, y_val in zip(X, y):
        prediction = quantum_classifier(weights, x_val)
        # If prediction is >= 0, classify as +1; else -1.
        label_pred = 1.0 if prediction >= 0 else -1.0
        if label_pred == y_val:
            correct += 1
    return correct / len(y)

# ------------------------------
# 5. SET UP OPTIMIZER AND INITIALIZE WEIGHTS
# ------------------------------
# Initialize weights with shape (num_layers, n_qubits, 3)
weights = pnp.array(np.random.uniform(0, 2*np.pi, (num_layers, n_qubits, 3)), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=learning_rate)

# ------------------------------
# 6. TRAINING LOOP
# ------------------------------
# Lists for storing metrics per epoch.
train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []
test_loss_hist = []
test_acc_hist = []

# For adversarial training, record additional test metrics.
adv_test_loss_hist = []
adv_test_acc_hist = []

# Start training.
print("Starting training with type:", training_type)
for epoch in range(1, num_epochs+1):
    start_time = time.time()
    
    # Define function to update weights on training set.
    def train_step(w):
    # For adversarial training, perturb each training input.
        loss_vals = []
        # Rename x_val, y_val → x_sample, y_label
        for x_sample, y_label in zip(X_train, y_train):
            x_input = x_sample.copy()
            # Apply FGSM or BIM if requested
            if training_type == "fgsm":
                x_input = fgsm_attack(w, x_input, y_label, epsilon)
            elif training_type == "bim":
                x_input = bim_attack(w, x_input, y_label,
                                    epsilon, bim_iterations, bim_alpha)
            # Use y_label, not y_val
            loss_vals.append((quantum_classifier(w, x_input) - y_label)**2)
        return pnp.mean(pnp.array(loss_vals))

    
    # Update weights using full-batch gradient descent.
    weights, train_cost = opt.step_and_cost(train_step, weights)
    
    # Calculate training accuracy and loss (clean version).
    train_loss = cost(weights, X_train, y_train)
    train_acc = accuracy(weights, X_train, y_train)
    
    # Evaluate on validation set (clean inputs).
    val_loss = cost(weights, X_val, y_val)
    val_acc = accuracy(weights, X_val, y_val)
    
    # Evaluate on test set (clean inputs).
    test_loss = cost(weights, X_test, y_test)
    test_acc = accuracy(weights, X_test, y_test)
    
    # For adversarial training, also evaluate on adversarially perturbed test set.
    if training_type in ["fgsm", "bim"]:
        adv_losses = []
        adv_correct = 0
        # 1. Rename x_val, y_val → x_sample, y_label
        for x_sample, y_label in zip(X_test, y_test):
            if training_type == "fgsm":
                x_adv = fgsm_attack(weights, x_sample.copy(), y_label, epsilon)
            else:  # "bim"
                x_adv = bim_attack(weights, x_sample.copy(),
                                y_label, epsilon,
                                bim_iterations, bim_alpha)
            # 2. Use y_label, not y_val
            adv_losses.append((quantum_classifier(weights, x_adv) - y_label)**2)
            # 3. Rename pred → pred_label
            pred_label = 1.0 if quantum_classifier(weights, x_adv) >= 0 else -1.0
            if pred_label == y_label:
                adv_correct += 1

        adv_test_loss = pnp.mean(pnp.array(adv_losses))
        adv_test_acc  = adv_correct / len(y_test)
    else:
        adv_test_loss = None
        adv_test_acc  = None


    epoch_time = time.time() - start_time

    # Append metrics.
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    if training_type in ["fgsm", "bim"]:
        adv_test_loss_hist.append(adv_test_loss)
        adv_test_acc_hist.append(adv_test_acc)

    # Print metrics for the epoch.
    print(f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    print(f"  Test Loss (clean): {test_loss:.4f} | Test Acc (clean): {test_acc*100:.2f}%")
    if training_type in ["fgsm", "bim"]:
        print(f"  Adv Test Loss: {adv_test_loss:.4f} | Adv Test Acc: {adv_test_acc*100:.2f}%")

# ------------------------------
# 7. SAVE METRICS TO CSV FILES
# ------------------------------
# Determine metrics save path based on training type.
metrics_prefix = metrics_folder + os.sep

# Save metrics (as CSV files with one column, each row corresponding to an epoch).
pd.DataFrame({"train_loss": train_loss_hist}).to_csv(metrics_prefix + "training_loss.csv", index=False)
pd.DataFrame({"train_acc": train_acc_hist}).to_csv(metrics_prefix + "training_accuracy.csv", index=False)
pd.DataFrame({"val_loss": val_loss_hist}).to_csv(metrics_prefix + "validation_loss.csv", index=False)
pd.DataFrame({"val_acc": val_acc_hist}).to_csv(metrics_prefix + "validation_accuracy.csv", index=False)
pd.DataFrame({"test_loss": test_loss_hist}).to_csv(metrics_prefix + "test_loss.csv", index=False)
pd.DataFrame({"test_acc": test_acc_hist}).to_csv(metrics_prefix + "test_accuracy.csv", index=False)

if training_type in ["fgsm", "bim"]:
    pd.DataFrame({"adv_test_loss": adv_test_loss_hist}).to_csv(metrics_prefix + "adversarial_test_loss.csv", index=False)
    pd.DataFrame({"adv_test_acc": adv_test_acc_hist}).to_csv(metrics_prefix + "adversarial_test_accuracy.csv", index=False)

# ------------------------------
# 8. SAVE FINAL WEIGHTS
# ------------------------------
# Construct weight filename based on training_type and epsilon (if adversarial).
if training_type == "clean":
    weight_filename = f"angle_{num_layers}layers_clean.npy"
elif training_type == "fgsm":
    weight_filename = f"angle_{num_layers}layers_fgsm_{epsilon}.npy"
elif training_type == "bim":
    weight_filename = f"angle_{num_layers}layers_bim_{epsilon}.npy"
else:
    weight_filename = f"angle_{num_layers}layers_unknown.npy"
weights_save_path = os.path.join(weights_folder, weight_filename)
np.save(weights_save_path, weights)
print("Training complete. Weights saved to:", weights_save_path)
