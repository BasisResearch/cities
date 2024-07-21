from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ZoningDataset(Dataset):
    def __init__(
        self,
        categorical,
        continuous,
        standardization_dictionary=None,
        index_dictionary=None,
    ):
        self.categorical = categorical
        self.continuous = continuous

        if index_dictionary is None:
            # this is hardcoded from data processing pipeline
            # and will be expanded in the future
            # for easier downstream use and interpretation
            self.index_dictionary = {
                "zoning_ordering": [
                    "downtown",
                    "blue_zone",
                    "yellow_zone",
                    "other_non_university",
                ],
                "limit_ordering": ["eliminated", "reduced", "full"],
            }

        self.standardization_dictionary = standardization_dictionary

        if self.categorical:
            self.categorical_levels = dict()
            for name in self.categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])
        
        N_categorical = len(categorical.keys())
        N_continuous = len(continuous.keys())

        if N_categorical > 0:
            self.n = len(next(iter(categorical.values())))
        elif N_continuous > 0:
            self.n = len(next(iter(continuous.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        cat_data = {key: val[idx] for key, val in self.categorical.items()}
        cont_data = {key: val[idx] for key, val in self.continuous.items()}
        return {
            "categorical": cat_data,
            "continuous": cont_data,
        }


def select_from_data(data, kwarg_names: Dict[str, List[str]]):
    _data = {}
    _data["outcome"] = data["continuous"][kwarg_names["outcome"]]
    _data["categorical"] = {
        key: val
        for key, val in data["categorical"].items()
        if key in kwarg_names["categorical"]
    }
    _data["continuous"] = {
        key: val
        for key, val in data["continuous"].items()
        if key in kwarg_names["continuous"]
    }

    return _data
