import os
import sys

import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import find_repo_root, standardize_and_scale
from cities.utils.data_grabber import list_available_features

sys.path.insert(0, os.path.dirname(os.getcwd()))


def test_data_folder():
    root = find_repo_root()
    folder_path = f"{root}/data/processed"
    file_names = os.listdir(folder_path)

    allowed_extensions = ["_wide.csv", "_long.csv", "_std_wide.csv", "_std_long.csv"]

    for file_name in file_names:
        if file_name != ".gitkeep":
            ends_with_allowed_extension = any(
                file_name.endswith(ext) for ext in allowed_extensions
            )
            assert (
                ends_with_allowed_extension
            ), f"File '{file_name}' does not have an allowed extension."

    all_features = list_available_features()
    for feature in all_features:
        valid_files = [
            feature + ext for ext in allowed_extensions if feature + ext in file_names
        ]
        assert (
            len(valid_files) == 4
        ), f"For feature '{feature}' some data formats are missing."


# set up gdp data
gdp = pd.read_csv("../data/raw/CAGDP1_2001_2021.csv", encoding="ISO-8859-1")

gdp = gdp.drop(gdp.columns[2:8], axis=1)
gdp = gdp.drop("2012", axis=1)
gdp.replace("(NA)", np.nan, inplace=True)
gdp.replace("(NM)", np.nan, inplace=True)
gdp.dropna(axis=0, inplace=True)

for column in gdp.columns[2:]:
    gdp[column] = gdp[column].astype(float)
# set up ends


def test_standardize_and_scale():
    gdp_scaled = standardize_and_scale(gdp)

    for column in gdp_scaled.columns[2:]:
        assert np.min(gdp_scaled[column]) >= -1
        assert np.max(gdp_scaled[column]) <= 1

    assert gdp.shape == gdp_scaled.shape


all_features = list_available_features()
assert "spending_commerce" in all_features
assert ".gitkeep" not in all_features
unique_features = []
for item in all_features:
    if item not in unique_features:
        unique_features.append(item)

assert len(unique_features) == len(all_features)
