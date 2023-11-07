from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def check_if_tensed(df):
    years_to_check = ["2015", "2018", "2019", "2020"]
    check = df.columns[2:].isin(years_to_check).any().any()
    return check
