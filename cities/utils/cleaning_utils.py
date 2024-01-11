from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def sigmoid(x, scale=1 / 3):
    range_0_1 = 1 / (1 + np.exp(-x * scale))
    range_minus1_1 = 2 * range_0_1 - 1
    return range_minus1_1


def standardize_and_scale(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes and scales float columns in a DataFrame to [-1,1], copying other columns. Returns a new DataFrame.
    """
    standard_scaler = StandardScaler()  # Standardize to mean 0, std 1

    # Copy all columns first
    new_data = data.copy()

    # Select float columns
    float_cols = data.select_dtypes(include=["float64"])

    # Standardize float columns to mean 0, std 1
    standardized_floats = standard_scaler.fit_transform(float_cols)

    # Apply sigmoid transformation, [-3std, 3std] to [-1, 1]
    new_data[float_cols.columns] = sigmoid(standardized_floats, scale=1 / 3)

    return new_data


def check_if_tensed(df):
    years_to_check = ["2015", "2018", "2019", "2020"]
    check = df.columns[2:].isin(years_to_check).any().any()
    return check
