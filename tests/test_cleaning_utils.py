import os
import sys

import numpy as np
import pandas as pd

from cities.utils.clean_variable import VariableCleaner
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import find_repo_root, list_available_features

sys.path.insert(0, os.path.dirname(os.getcwd()))

root = find_repo_root()
folder_paths = [f"{root}/data/processed", f"{root}/data/MSA_level"]


def test_data_folder():
    for folder in folder_paths:
        file_names = os.listdir(folder)

        allowed_extensions = [
            "_wide.csv",
            "_long.csv",
            "_std_wide.csv",
            "_std_long.csv",
        ]
        for file_name in file_names:
            if file_name != ".gitkeep":
                ends_with_allowed_extension = any(
                    file_name.endswith(ext) for ext in allowed_extensions
                )
                assert (
                    ends_with_allowed_extension
                ), f"File '{file_name}' does not have an allowed extension."

        all_features = []
        if folder.endswith("processed"):
            all_features = list_available_features()
        elif folder.endswith("MSA_level"):
            all_features = list_available_features("msa")

        for feature in all_features:
            valid_files = [
                feature + ext
                for ext in allowed_extensions
                if feature + ext in file_names
            ]
            assert (
                len(valid_files) == 4
            ), f"For feature '{feature}' some data formats are missing."


# set up gdp data
gdp = pd.read_csv(f"{root}/data/raw/CAGDP1_2001_2021.csv", encoding="ISO-8859-1")

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

    assert not gdp_scaled.isna().any().any()


def test_features_presence():
    all_features = list_available_features()
    all_features_msa = list_available_features("msa")
    all_features.extend(all_features_msa)

    assert "spending_commerce" in all_features
    assert ".gitkeep" not in all_features
    unique_features = []
    for item in all_features:
        if item not in unique_features:
            unique_features.append(item)

    assert len(unique_features) == len(all_features)


def test_variable_cleaner_drop_nans():
    data = {
        "GeoFIPS": [1, 2, 3],
        "GeoName": ["New York", np.nan, "Chicago"],
        "Value": [100, np.nan, 300],
    }
    df = pd.DataFrame(data)

    cleaner = VariableCleaner("variable", "path/to/raw.csv")

    cleaner.variable_df = df
    cleaner.drop_nans()

    assert len(cleaner.variable_df) == 2
    assert set(cleaner.variable_df.columns) == {"GeoFIPS", "GeoName", "Value"}
    assert not cleaner.variable_df["GeoName"].isna().any()
    assert not cleaner.variable_df["Value"].isna().any()
