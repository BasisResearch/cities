import os
import sys
from typing import List

import pandas as pd

from cities.utils.cleaning_utils import (find_repo_root,
                    check_if_tensed, list_available_features)

class DataGrabber:
    def __init__(self):
        self.repo_root = find_repo_root()
        sys.path.insert(0, self.repo_root)
        self.wide = {}
        self.std_wide = {}
        self.long = {}
        self.std_long = {}

    def get_features_wide(self, features: List[str]) -> None:
        for feature in features:
            file_path = os.path.join(
                self.repo_root, f"data/processed/{feature}_wide.csv"
            )
            self.wide[feature] = pd.read_csv(file_path)

    def get_features_std_wide(self, features: List[str]) -> None:
        for feature in features:
            file_path = os.path.join(
                self.repo_root, f"data/processed/{feature}_std_wide.csv"
            )
            self.std_wide[feature] = pd.read_csv(file_path)

    def get_features_long(self, features: List[str]) -> None:
        for feature in features:
            file_path = os.path.join(
                self.repo_root, f"data/processed/{feature}_long.csv"
            )
            self.long[feature] = pd.read_csv(file_path)

    def get_features_std_long(self, features: List[str]) -> None:
        for feature in features:
            file_path = os.path.join(
                self.repo_root, f"data/processed/{feature}_std_long.csv"
            )
            self.std_long[feature] = pd.read_csv(file_path)


def list_available_features():
    root = find_repo_root()
    folder_path = f"{root}/data/processed"
    file_names = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
    processed_file_names = []

    for file_name in file_names:
        # Use regular expressions to find the patterns and split accordingly
        matches = re.split(r"_wide|_long|_std", file_name)
        if matches:
            processed_file_names.append(matches[0])

    feature_names = list(set(processed_file_names))

    return feature_names


def list_tensed_features():
    data = DataGrabber()
    all_features = list_available_features()
    data.get_features_wide(all_features)
    
    tensed_features = []
    for feature in all_features:
        if check_if_tensed(data.wide[feature]):
            tensed_features.append(feature)
    
    return tensed_features    

