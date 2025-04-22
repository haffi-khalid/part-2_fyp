# Effects of Quantum Embeddings on the Generalization of Adversarially Trained Quantum Classifiers

**Author**: Muhammad Haffi Khalid  
**Institution**: University of Birmingham  
**Date**: 2025  

---

## 📘 Overview

This project investigates the robustness and generalization capabilities of variational quantum classifiers (VQCs) under adversarial training using different quantum data embeddings. The classifier is trained and evaluated on the UCI Banknote Authentication dataset using three embedding strategies:

- **Angle Encoding**
- **Amplitude Encoding**
- **Data Re-uploading Encoding**

Two adversarial training methods are used to test robustness:

- **FGSM (Fast Gradient Sign Method)**
- **BIM (Basic Iterative Method)**

The project uses **PennyLane**, **PyTorch**, and **Matplotlib**, and supports both **CPU and GPU (CUDA)** execution via TorchLayer.

---

## 📁 Folder Structure

```text
quantum_fyp/
│
├── data/                      # Preprocessed datasets per embedding
│   ├── angle/
│   ├── amplitude/
│   ├── Reupload/
│
├── dataset/                  # Preprocessing scripts for each embedding
│
├── models/                   # Quantum circuit definitions per embedding
│
├── adversaries/              # FGSM and BIM attack methods (per embedding)
│
├── train/                    # Training scripts (TorchLayer-based)
│
├── Results/                  # Evaluation results and graphs
│   ├── fgsm/
│   └── bim/
│
├── weights/                  # Saved model weights (.npy files)
│
├── Gen_data/                 # Training logs and metrics (.csv)
│   ├── angle/
│   ├── amplitude/
│   ├── Reupload/
│
└── README.md                 # Project documentation
```

---

## ⚙️ Requirements

To install dependencies inside a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Main dependencies:**
- `pennylane`
- `torch`
- `matplotlib`
- `numpy`
- `pandas`

---

## 🚀 How to Train a Model

From inside your virtual environment:

### 🔹 Clean Training (e.g., Amplitude)

```bash
python train/train_amplitude.py --training_type clean
```

### 🔹 Adversarial Training (FGSM)

```bash
python train/train_amplitude.py --training_type fgsm --epsilon 0.1
```

### 🔹 Adversarial Training (BIM)

```bash
python train/train_amplitude.py --training_type bim --epsilon 0.1 --bim_alpha 0.03 --bim_iter 5
```

You can do the same for:
- `train_angle.py`
- `train_reupload.py`

---

## 📊 Evaluation: Adversarial Robustness

To test how classifiers perform under FGSM or BIM:

### 🔹 FGSM Evaluation

```bash
python Results/fgsm/generate_fgsm_robustness_subset.py
```

### 🔹 BIM Evaluation

```bash
python Results/bim/generate_bim_robustness_subset.py
```

Generates plots and CSVs of:
- Accuracy vs Epsilon
- Loss vs Epsilon
- Combined graphs for comparison

---

## 📦 Outputs

- All **trained model weights** are saved in `weights/`
- All **training metrics** are saved as `.csv` files in `Gen_data/{embedding}/{training_type}/`
- All **evaluation plots** are saved in `Results/{fgsm|bim}/`

---

## 📈 Sample Metrics Tracked

- Training Loss & Accuracy
- Validation Loss & Accuracy
- Clean Test Loss & Accuracy
- Adversarial Test Loss & Accuracy

---

## 🎓 Thesis Focus

This repository supports the final year research project investigating:

> How different quantum embeddings affect the adversarial robustness and generalization of quantum classifiers trained on binary classification problems under adversarial attacks.

---

## 📎 Citation

Based on concepts and architecture from:

- Sirui Lu et al., *"Quantum Adversarial Machine Learning"*, [arXiv:2001.00030](https://arxiv.org/abs/2001.00030)

---

## 🙏 Acknowledgments

- PennyLane Team  
- University of Birmingham CS Department  
- My final year project supervisor : Dr Sharu Theresa Jose

---
