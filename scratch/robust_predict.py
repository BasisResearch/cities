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
from pyro.infer.autoguide import AutoDiagonalNormal, init_to_value
# from pyro.infer.autoguide import AutoDelta as AutoGuide
from chirho.observational.handlers import condition
from utils import nonify_dict, map_subset_onto_obs
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import os
import time
from collections import namedtuple

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
from pyro.contrib.easyguide import easy_guide
from cities.utils.data_loader import select_from_data
import pyro.distributions as dist
import argparse

from chirho.indexed.ops import gather, IndexSet

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--num_outer_if_mc", type=int, default=1000)
args = parser.parse_args()

SMOKE = args.smoke
NUM_OUTER_IF_MC = args.num_outer_if_mc

pyro.clear_param_store()
pyro.settings.set(module_local_params=True)

root = find_repo_root()


# <Fake AutoDelta Workaround>
def build_fake_delta(model, *args, **kwargs):

    # TODO easier way to do this? Also 'data' is a plate...
    with pyro.poutine.trace() as tr:
        model()
    latent_keys = [k for k, v in tr.get_trace().nodes.items() if not v['is_observed'] and k != 'data']

    latent_dim = 0
    for k in latent_keys:
        latent_dim += tr.get_trace().nodes[k]['value'].numel()

    params = torch.nn.Parameter(torch.zeros(latent_dim))

    @easy_guide(model)
    def fake_delta(self):
        group = self.group(match=".*")
        return group.sample("latents", dist.Normal(params, 1e-8).to_event(1))

    fake_delta._parameters = {"latents_group": params}

    return fake_delta

# </Fake AutoDelta Workaround>

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

SUPER_CAT_LEVELS = ct_dataset_read.categorical_levels
# </Data Load and Prep>


# <Unconditioned Model>
class UnconditionedTractsModel(TractsModel):
    def __init__(self, subset, n):
        self.subset = subset
        self.nonified_subset = nonify_dict(subset)
        super().__init__(
            **self.subset,
            categorical_levels=SUPER_CAT_LEVELS
        )

        self.n = n

    # noinspection PyMethodOverriding
    def forward(self):
        return super().forward(n=self.n, **self.nonified_subset)
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

# Passing the nonified subset to the forward ensures that all obs= get set to None.
full_unconditioned_model = UnconditionedTractsModel(subset, n=full_n)

# DEBUG
# full_unconditioned_model()

with pyro.plate("outer", size=13, dim=-2):
    result = full_unconditioned_model()
# Sanity check.
assert result.shape == (13, 816)

with pyro.plate("outer", size=14, dim=-3):
    result = full_unconditioned_model()
# Sanity check.
assert result.shape == (14, 1, 816)
# /DEBUG

full_conditioned_model = condition(
    full_unconditioned_model,
    data=subset_as_obs
)
# </Conditioned Model>

# <DEBUG check fake delta>
build_fake_delta(full_conditioned_model)
# </DEBUG check fake delta>

# <Training>
LR = 0.0025
NUM_STEPS = 100 if SMOKE else 2000
guide = run_svi_inference(
    full_conditioned_model,
    n_steps=NUM_STEPS,
    lr=LR,
    # vi_family=AutoGuide,
    guide=build_fake_delta(full_conditioned_model),
    plot=True
)

# </Training>

# <Generate Data>
# Construct the dataset sizes.
# sizes = [100, 200, 300, 500, 900]
# if not SMOKE:
#     sizes.extend([1200, 1900, 2700, 5000, 7500, 10000])
# sizes = list(range(100, 3000, 200))
sizes = [int(100 * 1.1 ** i) for i in range(2 if SMOKE else 50)]
# sizes = [100, 200]

datasets = []
for size in sizes:
    unconditioned_model_of_size = UnconditionedTractsModel(subset, n=size)
    predictive = Predictive(
        model=unconditioned_model_of_size,
        guide=guide,
        num_samples=1,
    )
    predictions = predictive()
    dataset = {k: predictions[k].squeeze() for k in subset_as_obs.keys()}
    datasets.append(dataset)
    print(f"Generated dataset of size {size}")
# </Generate Data>

# # <Sanity Check>
#
# # Compare seaborn density plots of the original (subset_as_obs) and generated data.
# comparison_dataset = datasets[-1]
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
    def __init__(self, model, num_tracts, num_mc=100):
        super().__init__()
        self.model = model
        # Keeping it simple for validation purposes. Removing limits entirely vs. requiring limits.
        self.no_limit = torch.zeros(num_tracts)
        self.full_limit = torch.ones(num_tracts)
        self.num_mc = num_mc

    def forward(self):
        with MultiWorldCounterfactual() as mwc:
            with do(actions={"limit": (self.no_limit, self.full_limit)}):
                with pyro.plate("mc", self.num_mc, dim=-2):
                    hu = self.model()

            hu0 = gather(hu, IndexSet(limit={1}, event_dim=0))
            hu1 = gather(hu, IndexSet(limit={2}, event_dim=0))

        # We want hu0 first, as that's the one we hypothesize will induce more units. This is the effect of
        #  removing the limit.
        return (hu0 - hu1).mean(dim=(-2, -1), keepdim=True).squeeze()
# </Functional>


# <Ground Truth>
ground_truth_estimator = TargetFunctional(
    PredictiveModel(full_unconditioned_model, guide),
    num_tracts=full_n,
    num_mc=100 if SMOKE else 10000
)
ground_truth_est = ground_truth_estimator()
ground_truth = -guide()[1]["weight_continuous_limit_housing_units"]
print(f"Estimated Ground Truth via Functional: {ground_truth_est.item()}")
print(f"Ground Truth Effect", ground_truth.item())
# </Ground Truth>


CorrectedEstimates = namedtuple("CorrectedEstimates", ["plugin", "plugin_actual", "corrected", "corrected_actual"])


def hack_fake_delta(model, fake_delta):
    # Because for some reason the fake delta isn't diffable in the context of the influence function
    #  machinery.
    return AutoDiagonalNormal(
        model,
        init_loc_fn=init_to_value(
            values=fake_delta()[1]
        ),
        init_scale=1e-8,
    )


def get_correction(data: Dict[str, torch.Tensor]):

    print(f"Preparing to process dataset of size {next(iter(data.values())).shape[0]}")

    # Split the data into train and correction.
    n = next(iter(data.values())).shape[0]
    train_n = n // 2
    correction_n = n - train_n

    train_data = {k: v[:train_n] for k, v in data.items()}
    correction_data = {k: v[train_n:] for k, v in data.items()}

    unconditioned_train_model = UnconditionedTractsModel(subset, n=train_n)
    unconditioned_correction_model = UnconditionedTractsModel(subset, n=correction_n)

    # Find the MAP.
    conditioned_model = condition(
        unconditioned_train_model,
        data=train_data
    )

    guide = run_svi_inference(
        conditioned_model,
        n_steps=NUM_STEPS * 2,
        lr=LR/2.,
        guide=build_fake_delta(conditioned_model),
        plot=False
    )
    guide = hack_fake_delta(conditioned_model, guide)

    model_for_plugin = PredictiveModel(unconditioned_train_model, guide)
    model_for_correction = PredictiveModel(unconditioned_correction_model, guide)

    functional_for_plugin = partial(
        TargetFunctional,
        num_tracts=train_n,
        num_mc=10 if SMOKE else 50000
    )
    functional_for_correction = partial(
        TargetFunctional,
        num_tracts=correction_n,
        num_mc=10 if SMOKE else 50000
    )

    estimator_for_plugin = functional_for_plugin(model_for_plugin)
    plugin_estimate = estimator_for_plugin()

    influence_functional = influence_fn(
        functional_for_correction,
        correction_data,
        pointwise_influence=False
    )

    correction_estimator = influence_functional(model_for_correction)
    with torch.no_grad():
        with MonteCarloInfluenceEstimator(
            num_samples_outer=10 if SMOKE else NUM_OUTER_IF_MC,
            num_samples_inner=1,
        ):
            correction = correction_estimator()

    plugin_actual = -guide()['weight_continuous_limit_housing_units']

    print(f"Parametric Effect: {plugin_actual.item()}")
    print(f"Plugin Estimate: {plugin_estimate.item()}")
    print(f"Correction: {correction.item()}")
    print()

    return CorrectedEstimates(
        plugin=plugin_estimate.detach(),
        plugin_actual=plugin_actual.detach(),
        corrected=(plugin_estimate + correction).detach(),
        corrected_actual=(plugin_actual + correction).detach()
    )
# <Correction>


plugins_and_corrections = []
for dataset in datasets:
    plugins_and_corrections.append(get_correction(dataset))

    # Save the results.
    with open(f"robust_predict_results{NUM_OUTER_IF_MC}.pt", "wb") as f:
        dill.dump(plugins_and_corrections, f)

# </Correction>
