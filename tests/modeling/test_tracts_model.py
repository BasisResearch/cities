import copy
import os

import pyro
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do
from pyro.infer import Predictive
from torch.utils.data import DataLoader

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


def test_tracts_model():

    # instantiate model simulate forward check shape
    tracts_model = TractsModel(
        **subset, categorical_levels=ct_dataset_read.categorical_levels
    )

    with pyro.poutine.trace() as tr:
        tracts_model(**subset)

    assert tr.trace.nodes["housing_units"]["value"].shape == torch.Size([816])

    # test inference
    pyro.clear_param_store()
    guide = run_svi_inference(
        tracts_model, n_steps=n_steps, lr=0.03, plot=False, **subset
    )

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
