import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_repo_root():
    """
    Finds the repo root (fodler containing .gitignore) and adds it to sys.path.
    """
    current_dir = os.getcwd()
    while True:
        marker_file_path = os.path.join(current_dir, ".gitignore")
        if os.path.isfile(marker_file_path):
            return current_dir

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    return current_dir


def standardize_and_scale(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes and scales float columns in a DataFrame to [-1,1], copying other columns. Returns a new DataFrame.
    """
    standard_scaler = StandardScaler()

    new_data = pd.DataFrame()
    for column in data.columns:
        if data.dtypes[column] != "float64":
            new_data[column] = data[column].copy()
        else:
            new = data[column].copy().values.reshape(-1, 1)
            new = standard_scaler.fit_transform(new)

            positive_mask = new >= 0
            negative_mask = new < 0

            min_positive = np.min(new[positive_mask])
            max_positive = np.max(new[positive_mask])
            scaled_positive = (new[positive_mask] - min_positive) / (
                max_positive - min_positive
            )

            min_negative = np.min(new[negative_mask])
            max_negative = np.max(new[negative_mask])
            scaled_negative = (new[negative_mask] - min_negative) / (
                max_negative - min_negative
            ) - 1

            scaled_values = np.empty_like(new, dtype=float)
            scaled_values[positive_mask] = scaled_positive
            scaled_values[negative_mask] = scaled_negative

            new_data[column] = scaled_values.reshape(-1)

    return new_data
