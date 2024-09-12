import copy
import os
from contextlib import ExitStack
from typing import List

import pyro
import pytest
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do
from pyro.infer import Predictive
from torch.utils.data import DataLoader

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel
from cities.modeling.zoning_models.zoning_tracts_sqm_model import TractsModelSqm
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

root = find_repo_root()

n_steps = 10
num_samples = 10


data_path = os.path.join(root, "data/minneapolis/processed/pg_census_tracts_dataset.pt")

dataset_read = torch.load(data_path)


loader = DataLoader(dataset_read, batch_size=len(dataset_read), shuffle=True)

data = next(iter(loader))


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
        "parcel_mean_sqm",
        "parcel_median_sqm",
        "parcel_sqm",
    },
    "outcome": "housing_units",
}

pg_subset = select_from_data(data, kwargs)
pg_dataset_read = torch.load(data_path)

print("shape for pg", pg_subset["categorical"]["year"].shape)


@pytest.mark.parametrize(
    "model_class",
    [TractsModel, TractsModelSqm],
)
def test_tracts_model(model_class):

    tracts_model = model_class(
        **pg_subset, categorical_levels=pg_dataset_read.categorical_levels
    )

    pyro.clear_param_store()
    guide = run_svi_inference(
        tracts_model, n_steps=n_steps, lr=0.03, plot=False, **pg_subset
    )

    with pyro.poutine.trace() as tr:
        units = tracts_model(**pg_subset)

    n = units.shape[0]

    assert tr.trace.nodes["housing_units"]["value"].shape == torch.Size([n])

    predictive = Predictive(tracts_model, guide=guide, num_samples=num_samples)

    subset_for_preds = copy.deepcopy(pg_subset)
    subset_for_preds["continuous"]["housing_units"] = None

    preds = predictive(**subset_for_preds)

    assert preds["housing_units"].shape == torch.Size([num_samples, n])

    # test counterfactuals
    with MultiWorldCounterfactual():
        with do(actions={"limit": (torch.tensor(0.0), torch.tensor(1.0))}):
            samples = predictive(**subset_for_preds)

    assert samples["housing_units"].shape == torch.Size([num_samples, 3, 1, 1, 1, n])


def context_stack(contexts: List):

    stack = ExitStack()
    for context in contexts:
        stack.enter_context(context)
    return stack


def assert_no_repeated_non_ones(tensor_size):

    non_ones = [dim for dim in tensor_size if dim != 1]
    assert len(non_ones) == len(
        set(non_ones)
    ), f"Repeated non-1 dimensions found in {tensor_size}"


@pytest.mark.parametrize(
    "use_plate, use_mwc, use_do",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
@pytest.mark.parametrize(
    "model_class",
    [TractsModel, TractsModelSqm],
)
def test_plated_sample_shaping(use_plate, use_mwc, use_do, model_class):
    model = model_class(**data, categorical_levels=pg_dataset_read.categorical_levels)

    # run forward to g
    units = model(**data)
    n = units.shape[0]
    print(n)

    pyro.clear_param_store()
    guide = run_svi_inference(model, n_steps=n_steps, lr=0.03, plot=False, **data)

    predictive = Predictive(model, guide=guide, num_samples=num_samples)

    data_for_preds = copy.deepcopy(data)
    data_for_preds["continuous"]["housing_units"] = None
    data_no_categorical_info = copy.deepcopy(data_for_preds)
    data_no_categorical_info["categorical"]["year"] = None

    context_managers = []

    if use_plate:
        context_managers.append(pyro.plate("outer_plate", 4, dim=-8))
    if use_mwc:
        context_managers.append(MultiWorldCounterfactual())
    if use_do:
        context_managers.append(
            do(actions={"limit": (torch.tensor(0.0), torch.tensor(1.0))})
        )

    with context_stack(context_managers):
        samples_contextualized = predictive(**data_for_preds)
        samples_conditioned_contextualized = predictive(**data)
        samples_no_categorical_info = predictive(**data_no_categorical_info)

    assert samples_conditioned_contextualized["housing_units"].shape == torch.Size(
        [num_samples, n]
    )

    assert_no_repeated_non_ones(samples_contextualized["housing_units"].shape)
    assert_no_repeated_non_ones(samples_no_categorical_info["housing_units"].shape)
