import os

import torch

from cities.utils.data_grabber import find_repo_root

import os

import dill

import matplotlib.pyplot as plt
import torch
import pyro
import copy

import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

import torch
import time

import pandas as pd
from torch.utils.data import DataLoader

from chirho.indexed.ops import IndexSet, gather
import seaborn as sns

import copy

import pyro
from pyro.infer import Predictive

from chirho.counterfactual.handlers import MultiWorldCounterfactual

from cities.modeling.zoning_models.zoning_tracts_model_ratios  import TractsModel
from cities.modeling.svi_inference import run_svi_inference
from cities.utils.data_loader import ZoningDataset,  select_from_data


from pyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    init_to_mean,
)



from cities.modeling.svi_inference import run_svi_inference
from pyro.infer import Predictive
from chirho.interventional.handlers import do


smoke_test = "CI" in os.environ

# use when testing model health
smoke_test = True

n_steps = 10 if smoke_test else 1500
num_samples = 10 if smoke_test else 1000

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


# from cities.utils.data_loader import select_from_data


# from cities.modeling.evaluation import (
#     prep_data_for_test,
#     test_performance,
# )


root = find_repo_root()


n_steps = 10
num_samples = 10


# data load and prep
census_tracts_data_path = os.path.join(
    root, "data/minneapolis/processed/census_tracts_dataset.pt"
)

ct_dataset_read = torch.load(census_tracts_data_path)

assert ct_dataset_read.n == 816

ct_loader = DataLoader(ct_dataset_read, batch_size=len(ct_dataset_read), shuffle=True)
data = next(iter(ct_loader))

kwargs = {
    "categorical": ["year", "census_tract"],
    "continuous": {
        "housing_units",
        "total_value",
        "median_value",
        "mean_limit_original",
        "median_distance",
        "income",
        "segregation_original",
        "white_original",
    },
    "outcome": "housing_units",
}
subset = select_from_data(data, kwargs)


# instantiate model simulate forward check shape
tracts_model = TractsModel(
    **subset, categorical_levels=ct_dataset_read.categorical_levels
)

with pyro.poutine.trace() as tr:
    tracts_model(**subset)

assert tr.trace.nodes["housing_units"]["value"].shape == torch.Size([816])

pyro.clear_param_store()
guide = run_svi_inference(tracts_model, n_steps=n_steps, lr=0.03, plot = False,
                           **subset)

predictive = Predictive(tracts_model, guide=guide, num_samples=num_samples)

subset_for_preds = copy.deepcopy(subset)
subset_for_preds["continuous"]["housing_units"] = None

preds = predictive(**subset_for_preds)
