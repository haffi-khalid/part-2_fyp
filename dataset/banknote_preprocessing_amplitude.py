# Script: banknote_preprocessing_amplitude.py

# Purpose:
#     This script preprocesses the Banknote Authentication dataset for amplitude encoding.
#     The main steps are:
#       1. Load the raw CSV file without a header, assigning custom column names.
#       2. Shuffle the DataFrame to randomize class order.
#       3. Split into train (70%), validation (15%), and test (15%) sets.
#       4. Normalize each sample (row) to unit Euclidean norm so that each feature
#          vector x satisfies ||x||₂ = 1, as required for amplitude encoding.
#       5. Save the resulting train/validation/test splits as CSV files in the
#          specified amplitude data folder.

# Inputs:
#     - Raw CSV file at:
#         C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv
#       Columns (no header) correspond to:
#         variance, skewness, curtosis, entropy, class

# Outputs:
#     - Preprocessed CSV files saved in:
#         C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude
#       Filenames:
#         * banknote_amplitude_preprocessed_train.csv
#         * banknote_amplitude_preprocessed_validation.csv
#         * banknote_amplitude_preprocessed_test.csv

# Author: Muhammad Haffi Khalid
# Date: [Today's Date]


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. File paths
input_file = r"C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
output_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\data\amplitude"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 2. Define column names and load dataset
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]
df = pd.read_csv(input_file, header=None, names=col_names)

# 3. Shuffle data
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Split into train (70%) and temp (30%)
df_train, df_temp = train_test_split(df_shuffled, test_size=0.30, random_state=42)

# Split temp into validation and test (each 15% of original)
df_validation, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# 5. Normalize each row to unit norm for amplitude encoding
feature_cols = ["variance", "skewness", "curtosis", "entropy"]

def normalize_row(row):
    vec = row.values.astype(float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return row  # or return zeros; dataset should not contain all-zero feature vectors
    return vec / norm

for split_df in (df_train, df_validation, df_test):
    # Apply normalization to feature columns, leave 'class' unchanged
    normalized_features = split_df[feature_cols].apply(normalize_row, axis=1, result_type="expand")
    normalized_features.columns = feature_cols
    split_df[feature_cols] = normalized_features

# 6. Save preprocessed splits
train_path = os.path.join(output_dir, "banknote_amplitude_preprocessed_train.csv")
val_path   = os.path.join(output_dir, "banknote_amplitude_preprocessed_validation.csv")
test_path  = os.path.join(output_dir, "banknote_amplitude_preprocessed_test.csv")

df_train.to_csv(train_path, index=False)
df_validation.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)

print("Amplitude encoding preprocessing complete.")
print(f"  Train:      {train_path}")
print(f"  Validation: {val_path}")
print(f"  Test:       {test_path}")
