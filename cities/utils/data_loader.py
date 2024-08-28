import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
import sqlalchemy
import pandas as pd


class ZoningDataset(Dataset):
    def __init__(
        self,
        categorical,
        continuous,
        standardization_dictionary=None,
    ):
        self.categorical = categorical
        self.continuous = continuous

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


def load_sql(kwargs):
    USERNAME = os.getenv("USERNAME")
    PASSWORD = os.getenv("PASSWORD")
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    engine = sqlalchemy.create_engine(
        f"postgresql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    )

    with engine.connect() as conn:
        dataset = pd.read_sql("select * from dev.census_tracts_wide", conn)
    dataset = {key: dataset[key].values for key in dataset.columns}

    return {
        "outcome": dataset[kwargs["outcome"]],
        "categorical": {
            key: torch.tensor(dataset[key]) for key in kwargs["categorical"]
        },
        "continuous": {
            key: torch.tensor(dataset[key], dtype=torch.float32)
            for key in kwargs["continuous"]
        },
    }
