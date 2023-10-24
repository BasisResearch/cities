import os
import sys

import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import (standardize_and_scale,
                                         list_available_features, find_repo_root)

sys.path.insert(0, os.path.dirname(os.getcwd()))


root = find_repo_root()
folder_path = f"{root}/data/processed"
file_names = os.listdir(folder_path)

allowed_extensions = ["_wide.csv", "_long.csv", "_std_wide.csv", "_std_long.csv"]

for file_name in file_names:
    if file_name != ".gitkeep":
        ends_with_allowed_extension = any(file_name.endswith(ext) for ext in allowed_extensions)
        assert ends_with_allowed_extension, f"File '{file_name}' does not have an allowed extension."



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



available_features = list_available_features()
assert "spending_commerce" in available_features
assert ".gitkeep" not in available_features
unique_features = []
for item in available_features:
    if item not in unique_features:
        unique_features.append(item)

assert len(unique_features) == len(available_features)