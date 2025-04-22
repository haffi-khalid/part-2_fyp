# Script: banknote_preprocessing_zzfm.py

# Purpose:
#     Preprocess the Banknote Authentication dataset for the ZZ-Feature Map (ZZFM).
#     Steps:
#       1. Load raw CSV (no header) and assign column names.
#       2. Shuffle and split into 70% train, 15% validation, 15% test.
#       3. Min–max scale each feature to [0, π] for use in RZ and ZZ rotations.
#       4. Save the splits as CSV files.

# Inputs:
#     Raw file:
#       C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv

# Outputs:
#     Train/Val/Test CSVs in:
#       C:\Users\ASDF\Desktop\part-2_fyp\data\ZZ-fm
#     Filenames:
#       * banknote_zzfm_preprocessed_train.csv
#       * banknote_zzfm_preprocessed_validation.csv
#       * banknote_zzfm_preprocessed_test.csv

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. Paths
input_csv = r"C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
output_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\data\ZZ-fm"
os.makedirs(output_dir, exist_ok=True)

# 2. Load and name columns
cols = ["variance", "skewness", "curtosis", "entropy", "class"]
df = pd.read_csv(input_csv, header=None, names=cols)

# 3. Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Split 70/15/15
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
df_val,   df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# 5. Min–max scale features to [0, π]
feature_cols = ["variance", "skewness", "curtosis", "entropy"]
for split in (df_train, df_val, df_test):
    X = split[feature_cols].values.astype(float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    # Avoid division by zero
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    X_scaled = (X - mins) / ranges * np.pi
    split[feature_cols] = X_scaled

# 6. Save CSVs
df_train.to_csv(os.path.join(output_dir, "banknote_zzfm_preprocessed_train.csv"), index=False)
df_val.to_csv(  os.path.join(output_dir, "banknote_zzfm_preprocessed_validation.csv"), index=False)
df_test.to_csv( os.path.join(output_dir, "banknote_zzfm_preprocessed_test.csv"), index=False)

print("ZZFM preprocessing complete:")
print(" Train CSV:     ", os.path.join(output_dir, "banknote_zzfm_preprocessed_train.csv"))
print(" Validation CSV:", os.path.join(output_dir, "banknote_zzfm_preprocessed_validation.csv"))
print(" Test CSV:      ", os.path.join(output_dir, "banknote_zzfm_preprocessed_test.csv"))
