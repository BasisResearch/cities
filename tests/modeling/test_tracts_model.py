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

from cities.modeling.evaluation import prep_data_for_test
from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

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

train_loader, test_loader, categorical_levels = prep_data_for_test(
    census_tracts_data_path, train_size=0.6
)


def test_tracts_model():

    tracts_model = TractsModel(
        **subset, categorical_levels=ct_dataset_read.categorical_levels
    )

    pyro.clear_param_store()
    guide = run_svi_inference(
        tracts_model, n_steps=n_steps, lr=0.03, plot=False, **subset
    )

    with pyro.poutine.trace() as tr:
        tracts_model(**subset)

    assert tr.trace.nodes["housing_units"]["value"].shape == torch.Size([816])

    predictive = Predictive(tracts_model, guide=guide, num_samples=num_samples)

    subset_for_preds = copy.deepcopy(subset)
    subset_for_preds["continuous"]["housing_units"] = None

    preds = predictive(**subset_for_preds)

    assert preds["housing_units"].shape == torch.Size([num_samples, 816])

    # test counterfactuals
    with MultiWorldCounterfactual():
        with do(actions={"limit": (torch.tensor(0.0), torch.tensor(1.0))}):
            samples = predictive(**subset_for_preds)

    assert samples["housing_units"].shape == torch.Size([num_samples, 3, 1, 1, 1, 816])


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
    "model_class, data",
    [
        (TractsModel, subset),
    ],
)
def test_plated_sample_shaping(use_plate, use_mwc, use_do, model_class, data):

    model = model_class(**data, categorical_levels=ct_dataset_read.categorical_levels)

    # run forward to g
    units = model(**data)
    n = units.shape[0]

    pyro.clear_param_store()
    guide = run_svi_inference(model, n_steps=n_steps, lr=0.03, plot=False, **data)

    predictive = Predictive(model, guide=guide, num_samples=num_samples)

    data_for_preds = copy.deepcopy(data)
    data_for_preds["continuous"]["housing_units"] = None

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

    # print("plates used: ", use_plate, use_mwc, use_do)
    # print(samples_contextualized['housing_units'].shape)
    # print(samples_conditioned_contextualized['housing_units'].shape)

    assert samples_conditioned_contextualized["housing_units"].shape == torch.Size(
        [num_samples, n]
    )

    assert_no_repeated_non_ones(samples_contextualized["housing_units"].shape)


# test_plated_sample_shaping(True, True, True, TractsModel, subset)
# test_plated_sample_shaping(True, True, False, TractsModel, subset)
# test_plated_sample_shaping(True, False, True, TractsModel, subset)
# test_plated_sample_shaping(False, True, True, TractsModel, subset)
