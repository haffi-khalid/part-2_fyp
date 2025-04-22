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

## 📦 Requirements

To get started, install the required packages:

```bash
pip install pennylane numpy pandas matplotlib scikit-learn
```

> ✅ Tested with Python 3.11 and PennyLane ≥ 0.32.

---

## 🧠 Embeddings Overview

| Embedding    | Qubits Used | Encoding Strategy         |
|--------------|-------------|---------------------------|
| **Angle**    | 4           | RY rotations (min–max [0, π]) |
| **Amplitude**| 3 (2 + 1)   | L2-normalized row-wise amplitude encoding |
| **Re-upload**| 5 (4 + 1)   | RY with data re-uploading at each layer |
| **ZZ-FM**    | TBD         | Feature map-based (reserved) |

---

## 🗂️ Folder Structure

<details>
<summary>Click to expand</summary>

```
part-2_fyp/
├── bim_robustness.py
├── generate_fgsm_robustness.py
├── generate_bim_robustness.py

├── adversaries/
│   ├── adversaries_{angle, amplitude, reupload, zzfm}.py

├── data/
│   ├── data_banknote_authentication.csv
│   └── {angle, amplitude, Reupload, ZZ-fm}/
│       └── banknote_*_preprocessed_{train|val|test}.csv

├── dataset/
│   └── banknote_preprocessing_{angle, amplitude, reupload, zzfm}.py

├── Gen_data/
│   └── {angle, amplitude, Reupload, ZZfm}/
│       └── {clean, adversarial}/
│           └── *.csv (metrics, gradients, ⟨Z⟩ distributions)

├── models/
│   ├── quantum_model_{angle, amplitude, reupload, zzfm}.py

├── Results/
│   └── {fgsm, bim, bim_stronger}/
│       ├── *_metrics.csv
│       ├── *_acc_vs_eps.png, *_loss_vs_eps.png

├── train/
│   ├── train_{angle, amplitude, reupload, zzfm}.py

└── weights/
    └── {angle, amplitude, Reupload, ZZfm}/
        └── *_10layers_{clean|fgsm_ε|bim_ε}.npy
```
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
## 🧪 Running the Project

### 🧹 Preprocess Dataset

```bash
cd dataset/
python banknote_preprocessing_angle.py
python banknote_preprocessing_amplitude.py
python banknote_preprocessing_reupload.py
```


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
- **Pérez-Salinas et al.**, *Quantum*, 2020 — [Data Re-uploading Classifier](https://quantum-journal.org/papers/q-2020-02-06-226/)
- **Goodfellow et al.**, *arXiv:1412.6572*, 2014 — [FGSM](https://arxiv.org/abs/1412.6572)


---

## 🙏 Acknowledgments

- PennyLane Team  
- University of Birmingham CS Department  
- My final year project supervisor : Dr Sharu Theresa Jose

---

