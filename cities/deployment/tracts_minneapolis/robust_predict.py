import os

import dill
import pyro
import torch
from torch.utils.data import DataLoader
from functools import partial

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

# can be disposed of once you access data in a different manner
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data
from copy import deepcopy
from pyro.infer.autoguide import AutoDelta
from chirho.observational.handlers import condition
from utils import nonify_dict, map_subset_onto_obs
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os
import time

import dill
import pandas as pd
import pyro
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual

# import chirho
from chirho.interventional.handlers import do
from pyro.infer import Predictive
from torch.utils.data import DataLoader

from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

from chirho.robust.handlers.estimators import MonteCarloInfluenceEstimator, one_step_corrected_estimator
from chirho.robust.ops import influence_fn
from chirho.observational.handlers.predictive import PredictiveModel

# can be disposed of once you access data in a different manner
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

from chirho.indexed.ops import gather, IndexSet

pyro.settings.set(module_local_params=True)

root = find_repo_root()

# <Data Load and Prep>
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
# </Data Load and Prep>

# <Unconditioned Model>
nonified_subset = nonify_dict(subset)

# Initialize the tracts model with the full dataset. This establishes the full categorical leveling.
_og_model = TractsModel(
    **subset,
    categorical_levels=ct_dataset_read.categorical_levels
)

# Passing the nonified subset to the forward ensures that all obs= get set to None.
unconditioned_model = partial(_og_model, **nonified_subset)
# </Unconditioned Model>

# <Conditioned Model>
# Convert the subset-formed dataset into something we can pass to obs for conditioning.
subset_as_obs = map_subset_onto_obs(
    subset,
    site_names=[
        "year",
        "distance",
        "white",
        "segregation",
        "income",
        "limit",
        "median_value",
        "housing_units"
    ]
)
full_n = next(iter(subset_as_obs.values())).shape[0]

full_conditioned_model = condition(
    partial(unconditioned_model, n=full_n),
    data=subset_as_obs
)
# </Conditioned Model>

# <Training>

guide = run_svi_inference(
    full_conditioned_model,
    n_steps=1000,
    lr=0.1,
    vi_family=AutoDelta,
    plot=False
)

# </Training>

# <Generate Data>
# Construct the dataset sizes.
sizes = [100, 300, 900]  # , 2700, 8100, 24300]

# Trace the latents for replay (we're using a delta, so only one is needed).
map_params = guide()

datasets = []
for size in sizes:
    predictive = Predictive(
        model=partial(unconditioned_model, n=size),
        guide=guide,
        num_samples=1,
    )
    predictions = predictive()
    dataset = {k: predictions[k].squeeze() for k in subset_as_obs.keys()}
    datasets.append(dataset)
# </Generate Data>

# # <Sanity Check>
#
# # Compare seaborn density plots of the original (subset_as_obs) and generated data.
# comparison_dataset = datasets[2]
# for (k, v1), v2 in zip(subset_as_obs.items(), comparison_dataset.values()):
#     plt.figure()
#     sns.kdeplot(v1, label="Real")
#     sns.kdeplot(v2, label="Synthetic")
#     plt.suptitle(f"{k} Comparison")
#     plt.legend()
# plt.show()
#
# # </Sanity Check>


# <Functional>
class TargetFunctional(torch.nn.Module):
    def __init__(self, model, guide, num_tracts, num_mc=100):
        super().__init__()
        self.predictive_model = Predictive(
            model=model,
            guide=guide,
            num_samples=num_mc
        )
        # Keeping it simple for validation purposes. Removing limits entirely vs. requiring limits.
        self.no_limit = torch.zeros(num_tracts)
        self.full_limit = torch.ones(num_tracts)

    def forward(self):
        with MultiWorldCounterfactual() as mwc:
            with do(actions={"limit": (self.no_limit, self.full_limit)}):
                hu = self.predictive_model()["housing_units"]

            hu0 = gather(hu, IndexSet(limit={1}, event_dim=0))
            hu1 = gather(hu, IndexSet(limit={2}, event_dim=0))

        # We want hu0 first, as that's the one we hypothesize will induce more units. This is the effect of
        #  removing the the limit.
        return (hu0 - hu1).mean(dim=0, keepdim=True).mean(dim=-1, keepdim=True).squeeze()

# </Functional>


# <Ground Truth>

ground_truth_functional = partial(
    TargetFunctional,
    guide=guide,
    num_tracts=full_n,
    num_mc=1000
)
ground_truth_estimator = ground_truth_functional(partial(unconditioned_model, n=full_n))
ground_truth = ground_truth_estimator()
print("GT Housing Unit Delta Between Full and No Limits: ", ground_truth.item())
# </Ground Truth>


def get_correction(unconditioned_model, data: Dict[str, torch.Tensor]):

    # Split the data into train and correction.
    n = next(iter(data.values())).shape[0]
    train_n = n // 2
    correction_n = n - train_n

    train_data = {k: v[:train_n] for k, v in data.items()}
    correction_data = {k: v[train_n:] for k, v in data.items()}

    # Find the MAP.
    conditioned_model = condition(
        partial(unconditioned_model, n=train_n),
        data=train_data
    )

    guide = run_svi_inference(
        conditioned_model,
        n_steps=1000,
        lr=0.1,
        vi_family=AutoDelta,
        plot=False
    )

    model_for_plugin = partial(unconditioned_model, n=train_n)
    model_for_correction = partial(unconditioned_model, n=correction_n)

    functional_for_plugin = partial(
        TargetFunctional,
        guide=guide,
        num_tracts=train_n
    )
    functional_for_correction = partial(
        TargetFunctional,
        guide=guide,
        num_tracts=correction_n
    )

    estimator_for_plugin = functional_for_plugin(model_for_plugin)
    plugin_estimate = estimator_for_plugin()

    influence_functional = influence_fn(
        functional_for_correction,
        correction_data,
        pointwise_influence=False
    )

    correction_estimator = influence_functional(model_for_correction)
    with MonteCarloInfluenceEstimator(
        num_samples_outer=1000,
        num_samples_inner=1,
    ):
        correction = correction_estimator()

    return plugin, plugin + correction


# <Correction>

plugins_and_corrections = []

for dataset in datasets:
    plugin, corrected = get_correction(unconditioned_model, dataset)
    plugins_and_corrections.append((plugin, corrected))

# </Correction>

# <Plot>

# Plot the plugin and correction agains the ground truth for each dataset size.
raise NotImplementedError()

# </Plot>


# TODO
# 1. [ ] pass in the whole (real) dataset to the initializer.
# 2. [ ] construct a partial of the with the nonified dataset and split


# Notes
# 1. train AutoDelta on real data
# 2. treat that as GT and generate arbitrarily large sets of data (100, 300, 900, 2700, 8100, 24300)
# 3. split data into train and correction
# 4. train AutoDelta on the train data
# 5. get correction wrt correction data
# 6. show GT functional, AutoDelta functional, and corrected functional
# 7. plot the difference between GT and AutoDelta, GT and corrected, over the dataset size on x-axis

pass