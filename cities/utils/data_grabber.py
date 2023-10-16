import os
import sys
from typing import Dict, List

import pandas as pd

from cities.utils.cleaning_utils import find_repo_root


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
