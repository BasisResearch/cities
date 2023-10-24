import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import re



def find_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


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


def list_available_features():
    
    root = find_repo_root()
    folder_path = f"{root}/data/processed"
    file_names = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
    processed_file_names = []

    for file_name in file_names:
        # Use regular expressions to find the patterns and split accordingly
        matches = re.split(r'_wide|_long|_std', file_name)
        if matches:
            processed_file_names.append(matches[0])
        
    feature_names = list(set(processed_file_names))

    return feature_names
