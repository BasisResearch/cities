from typing import List, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cities.utils.data_grabber import DataGrabber

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


def revert_standardize_and_scale_scaler(
    transformed_values: Union[np.ndarray, List, pd.Series, float],
    year: int,
    variable_name: str,
) -> List:
    if not isinstance(transformed_values, np.ndarray):
        transformed_values = np.array(transformed_values)

    def inverse_sigmoid(y, scale=1 / 3):
        return -np.log((2 / (y + 1)) - 1) / scale

    # needed to avoid lint issues
    dg: DataGrabber

    # normally this will be deployed in a context in which dg already exists
    # and we want to avoid wasting time by reloading the data
    try:
        original_column = dg.wide[variable_name][str(year)].values
    except NameError:
        dg = DataGrabber()
        dg.get_features_wide([variable_name])
        original_column = dg.wide[variable_name][str(year)].values.reshape(-1, 1)

    # dg = DataGrabber()
    # dg.get_features_wide([variable_name])

    # original_column = dg.wide[variable_name][str(year)].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(original_column)

    inverted_values_sigmoid = inverse_sigmoid(transformed_values)
    inverted_values = scaler.inverse_transform(
        inverted_values_sigmoid.reshape(-1, 1)
    ).flatten()

    return inverted_values


def revert_prediction_df(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    df_copy = df.copy()

    for i in range(len(df)):
        df_copy.iloc[i, 1:] = revert_standardize_and_scale_scaler(
            df.iloc[i, 1:].tolist(), df.iloc[i, 0], variable_name
        )

    return df_copy


def check_if_tensed(df):
    years_to_check = ["2015", "2018", "2019", "2020"]
    check = df.columns[2:].isin(years_to_check).any().any()
    return check