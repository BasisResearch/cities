import os
from typing import Dict, List

import pandas as pd
import sqlalchemy
import torch
from torch.utils.data import Dataset


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


def db_connection():
    DB_USERNAME = os.getenv("DB_USERNAME")
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    PASSWORD = os.getenv("PASSWORD")
    DB_SEARCH_PATH = os.getenv("DB_SEARCH_PATH")

    return sqlalchemy.create_engine(
        f"postgresql://{DB_USERNAME}:{PASSWORD}@{HOST}/{DATABASE}",
        connect_args={"options": f"-csearch-path={DB_SEARCH_PATH}"},
    ).connect()


def select_from_sql(sql, conn, kwargs, params=None):
    df = pd.read_sql(sql, conn, params=params)
    return {
        "outcome": df[kwargs["outcome"]],
        "categorical": {
            key: torch.tensor(df[key].values, dtype=torch.int64)
            for key in kwargs["categorical"]
        },
        "continuous": {
            key: torch.tensor(df[key], dtype=torch.float32)
            for key in kwargs["continuous"]
        },
    }
