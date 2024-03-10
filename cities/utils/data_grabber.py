import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def find_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def check_if_tensed(df):
    years_to_check = ["2015", "2018", "2019", "2020"]
    check = df.columns[2:].isin(years_to_check).any().any()
    return check


class DataGrabberCSV:
    def __init__(self):
        self.repo_root = find_repo_root()
        self.data_path = os.path.join(self.repo_root, "data/processed")
        self.wide: Dict[str, pd.DataFrame] = {}
        self.std_wide: Dict[str, pd.DataFrame] = {}
        self.long: Dict[str, pd.DataFrame] = {}
        self.std_long: Dict[str, pd.DataFrame] = {}

    def _get_features(self, features: List[str], table_suffix: str) -> None:
        for feature in features:
            file_path = os.path.join(self.data_path, f"{feature}_{table_suffix}.csv")
            df = pd.read_csv(file_path)
            if table_suffix == "wide":
                self.wide[feature] = df
            elif table_suffix == "std_wide":
                self.std_wide[feature] = df
            elif table_suffix == "long":
                self.long[feature] = df
            elif table_suffix == "std_long":
                self.std_long[feature] = df
            else:
                raise ValueError(
                    "Invalid table suffix. Please choose 'wide', 'std_wide', 'long', or 'std_long'."
                )

    def get_features_wide(self, features: List[str]) -> None:
        self._get_features(features, "wide")

    def get_features_std_wide(self, features: List[str]) -> None:
        self._get_features(features, "std_wide")

    def get_features_long(self, features: List[str]) -> None:
        self._get_features(features, "long")

    def get_features_std_long(self, features: List[str]) -> None:
        self._get_features(features, "std_long")


class MSADataGrabberCSV(DataGrabberCSV):
    def __init__(self):
        super().__init__()
        self.repo_root = find_repo_root()
        self.data_path = os.path.join(self.repo_root, "data/MSA_level")
        sys.path.insert(0, self.data_path)


def list_available_features(level="county"):
    root = find_repo_root()

    if level == "county":
        folder_path = f"{root}/data/processed"
    elif level == "msa":
        folder_path = f"{root}/data/MSA_level"
    else:
        raise ValueError("Invalid level. Please choose 'county' or 'msa'.")

    file_names = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
    processed_file_names = []

    for file_name in file_names:
        # Use regular expressions to find the patterns and split accordingly
        matches = re.split(r"_wide|_long|_std", file_name)
        if matches:
            processed_file_names.append(matches[0])

    feature_names = list(set(processed_file_names))

    return sorted(feature_names)


def list_tensed_features(level="county"):
    if level == "county":
        data = DataGrabber()
        all_features = list_available_features(level="county")

    elif level == "msa":
        data = MSADataGrabber()
        all_features = list_available_features(level="msa")

    else:
        raise ValueError("Invalid level. Please choose 'county' or 'msa'.")

    data.get_features_wide(all_features)

    tensed_features = []
    for feature in all_features:
        if check_if_tensed(data.wide[feature]):
            tensed_features.append(feature)

    return sorted(tensed_features)


# TODO this only will pick up spending-based interventions
# needs to be modified/expanded when we add other types of interventions
def list_interventions():
    interventions = [
        feature for feature in list_tensed_features() if feature.startswith("spending_")
    ]
    return sorted(interventions)


def list_outcomes():
    outcomes = [
        feature
        for feature in list_tensed_features()
        if feature not in list_interventions()
    ]
    return sorted(outcomes)


DataGrabber = DataGrabberCSV

MSADataGrabber = MSADataGrabberCSV
