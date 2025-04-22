
# Script: banknote_preprocessing_angle.py
# Purpose:
#     This script preprocesses the banknote authentication dataset for angle encoding.
#     The key steps include:
#       1. Loading the CSV file from the specified path using predefined column names.
#       2. Shuffling the data to ensure a random mix of samples.
#       3. Splitting the data into training (70%), validation (15%), and testing (15%) sets.
#       4. Normalizing the continuous features (variance, skewness, curtosis, entropy) using
#          min–max scaling to map values to the range [0, π] (for compatibility with angle encoding).
#       5. Saving the preprocessed train, validation, and test sets as CSV files in the designated folder.

# Inputs:
#     - CSV file located at "C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
#       (The file does not contain a header row; hence custom column names are provided.)

# Outputs:
#     - Preprocessed CSV files saved in "C:\Users\ASDF\Desktop\part-2_fyp\data" with the names:
#         * banknote_angle_preprocessed_train.csv
#         * banknote_angle_preprocessed_validation.csv
#         * banknote_angle_preprocessed_test.csv

# Methodology:
#     1. Load the CSV file using pandas with header=None and assign custom column names.
#     2. Shuffle the DataFrame using a fixed random state.
#     3. Split the data using a two-step process: first into training (70%) and a temporary set (30%),
#        then splitting the temporary set evenly into validation and testing sets (15% each).
#     4. Normalize each of the continuous feature columns (variance, skewness, curtosis, entropy)
#        to the range [0, π] using min–max scaling.
#     5. Save the processed DataFrames as CSV files in the specified output directory.

# Author: Muhammad Haffi Khalid
# Date: [Today's Date]

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the file paths for input and output.
input_file_path = r"C:\Users\ASDF\Desktop\part-2_fyp\data\data_banknote_authentication.csv"
output_dir = r"C:\Users\ASDF\Desktop\part-2_fyp\data"

# Ensure the output directory exists.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the column names based on the dataset description.
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

# 1. Load the dataset from the CSV file without a header (assign custom column names).
df = pd.read_csv(input_file_path, header=None, names=col_names)

# 2. Shuffle the data to ensure randomness.
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Split the data into training (70%) and a temporary set (30% for validation + test).
df_train, df_temp = train_test_split(df_shuffled, test_size=0.30, random_state=42)

# Now, split the temporary set evenly into validation and test sets (each 15% of original data).
df_validation, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# 4. Normalize the continuous features for angle encoding.
# The features to scale are: "variance", "skewness", "curtosis", "entropy"
def min_max_scale_to_pi(series):
    """
    Scales a pandas Series to the range [0, π] using min–max scaling.
    
    Formula:
        x_scaled = ((x - x_min) / (x_max - x_min)) * π
    """
    x_min = series.min()
    x_max = series.max()
    # Avoid division by zero if all values are the same
    if x_max == x_min:
        return series.apply(lambda x: np.pi / 2)
    return ((series - x_min) / (x_max - x_min)) * np.pi

# List of feature columns that require scaling.
feature_cols = ["variance", "skewness", "curtosis", "entropy"]

# Apply the scaling function to each feature column in all splits.
for df_split in [df_train, df_validation, df_test]:
    for col in feature_cols:
        df_split[col] = min_max_scale_to_pi(df_split[col])

# 5. Save the preprocessed data as CSV files in the output directory.
output_train = os.path.join(output_dir, "banknote_angle_preprocessed_train.csv")
output_validation = os.path.join(output_dir, "banknote_angle_preprocessed_validation.csv")
output_test = os.path.join(output_dir, "banknote_angle_preprocessed_test.csv")

df_train.to_csv(output_train, index=False)
df_validation.to_csv(output_validation, index=False)
df_test.to_csv(output_test, index=False)

print("Preprocessing complete. The preprocessed files are saved as:")
print(f"Train: {output_train}")
print(f"Validation: {output_validation}")
print(f"Test: {output_test}")
