# File: dataset_preprocessing_reupload.py
# Author: Muhammad Haffi Khalid
# Date: [Today's Date]
#
# Purpose:
#     Load the raw UCI Banknote Authentication CSV, shuffle and split into
#     70% train / 15% validation / 15% test, apply per-feature min–max
#     normalization into [0, π], and save the three preprocessed CSV files.
#
# Usage:
#     python dataset_preprocessing_reupload.py
#
# Output:
#     - banknote_reupload_preprocessed_train.csv
#     - banknote_reupload_preprocessed_validation.csv
#     - banknote_reupload_preprocessed_test.csv
#     all saved under C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload

import os
import pandas as pd
import numpy as np

# 1. Configure input and output paths
raw_csv = r"C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
out_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\data\Reupload"
os.makedirs(out_dir, exist_ok=True)

# 2. Load and shuffle the dataset
df = pd.read_csv(raw_csv, header=None,
                 names=["variance", "skewness", "curtosis", "entropy", "class"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Split into train / validation / test
n = len(df)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train:n_train + n_val]
test_df  = df.iloc[n_train + n_val:]

# 4. Per-feature min–max normalization to [0, π]
def scale_to_pi(df_input):
    df_out = df_input.copy()
    for col in ["variance", "skewness", "curtosis", "entropy"]:
        mn = df_out[col].min()
        mx = df_out[col].max()
        df_out[col] = (df_out[col] - mn) / (mx - mn) * np.pi
    return df_out

train_scaled = scale_to_pi(train_df)
val_scaled   = scale_to_pi(val_df)
test_scaled  = scale_to_pi(test_df)

# 5. Save the preprocessed CSV files
train_scaled.to_csv(os.path.join(out_dir, "banknote_reupload_preprocessed_train.csv"), index=False)
val_scaled.to_csv(  os.path.join(out_dir, "banknote_reupload_preprocessed_validation.csv"), index=False)
test_scaled.to_csv( os.path.join(out_dir, "banknote_reupload_preprocessed_test.csv"), index=False)

print("Re‑upload preprocessing complete.")
print(f"Saved train/val/test CSVs to: {out_dir}")
