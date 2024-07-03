import copy
import os

import pyro
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do
from chirho.robust.handlers.predictive import PredictiveModel
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.utils.data import DataLoader

from cities.modeling.add_causal_layer import AddCausalLayer
from cities.modeling.simple_linear import SimpleLinear
from cities.modeling.svi_inference import run_svi_inference
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

root = find_repo_root()

#######################
# set up sythentic data
#######################

n = 600
part = n // 3
n_steps = 600

x_cat = torch.cat(
    [
        torch.zeros([part], dtype=torch.long),
        torch.ones([part], dtype=torch.long),
        2 * torch.ones([part], dtype=torch.long),
    ]
)
x_con = (
    torch.cat([torch.ones(n // 2), torch.ones(n // 2) * 2])
    + torch.randn(n) * 0.1
    + 0.5 * x_cat
)

y_mixed = x_cat * 2 + x_con * 2 + torch.randn(n) * 0.05

synthetic_minimal_outcome = {
    "categorical": {"x_cat": x_cat},
    "continuous": {"x_con": x_con, "y_mixed": y_mixed},
    "outcome": y_mixed,
}

minimal_kwargs_synthetic = {
    "categorical": {"x_cat"},
    "continuous": {"x_con"},
    "outcome": "y_mixed",
}


def test_basic_layer_synthetic():

    pyro.clear_param_store()

    base_model_synthetic = SimpleLinear(**synthetic_minimal_outcome)

    def new_model_synthetic(**kwargs):
        with AddCausalLayer(
            base_model_synthetic,
            model_kwargs=minimal_kwargs_synthetic,
            dataset=synthetic_minimal_outcome,
            causal_layer={"x_con": ["x_cat"]},
        ):
            base_model_synthetic(**kwargs)

    new_guide_synthetic = run_svi_inference(
        new_model_synthetic, **synthetic_minimal_outcome, vi_family=AutoDiagonalNormal
    )
    # note AutoMultivariateNormal sometimes fails, not clear why

    predictive_synthetic = Predictive(
        new_model_synthetic, guide=new_guide_synthetic, num_samples=500, parallel=True
    )

    predictive_model_synthetic = PredictiveModel(
        new_model_synthetic, guide=new_guide_synthetic
    )

    synthetic_minimal = copy.deepcopy(synthetic_minimal_outcome)
    synthetic_minimal["outcome"] = None

    samples_synthetic = predictive_synthetic(**synthetic_minimal)

    w_x_cat_to_x_con = samples_synthetic["weights_categorical_x_cat_x_con"].squeeze()

    cat0_mean = w_x_cat_to_x_con[:, 0].mean()
    cat1_mean = w_x_cat_to_x_con[:, 1].mean()
    cat2_mean = w_x_cat_to_x_con[:, 2].mean()

    assert torch.allclose(
        torch.tensor([cat0_mean, cat1_mean, cat2_mean]),
        torch.tensor([1.0, 2.0, 3.0]),
        atol=0.2,
    )

    with MultiWorldCounterfactual():
        with do(actions={"weights_categorical_x_cat": torch.tensor([10.0]).expand(3)}):
            with pyro.poutine.trace() as tr_after_mwc:
                predictive_model_synthetic(
                    categorical={"x_cat": x_cat},
                    continuous={"x_con": x_con},
                    outcome=None,
                )

    assert torch.allclose(
        tr_after_mwc.trace.nodes["weights_categorical_x_cat"]["value"]
        .detach()
        .squeeze()[1, :],
        torch.tensor([10.0]),
    )

    outcome_values = tr_after_mwc.trace.nodes["outcome_observed"]["value"].squeeze()

    before = outcome_values[0, :].detach().numpy()
    after = outcome_values[1, :].detach().numpy()
    assert (after - before > 5).all()


###############################


# real data
zoning_data_path = os.path.join(root, "data/minneapolis/processed/zoning_dataset.pt")
zoning_dataset_read = torch.load(zoning_data_path)

zoning_loader = DataLoader(
    zoning_dataset_read, batch_size=len(zoning_dataset_read), shuffle=True
)

data = next(iter(zoning_loader))

minimal_kwargs = {
    "categorical": ["past_reform"],
    "continuous": {"parcel_area"},
    "outcome": "housing_units",
}

minimal_subset = select_from_data(data, minimal_kwargs)

expanded_kwargs = {
    "categorical": ["zone_id", "past_reform"],
    "continuous": {"parcel_area"},
    "outcome": "housing_units",
}

expanded_subset = select_from_data(data, expanded_kwargs)

###############################
# tests with real data
################################


def test_basic_layer():

    base_model = SimpleLinear(**minimal_subset)

    def new_model(**kwargs):
        with AddCausalLayer(
            base_model,
            model_kwargs=minimal_kwargs,
            dataset=data,
            causal_layer={
                "parcel_area": ["zone_id", "neighborhood_id", "car_parking_original"],
                "housing_units": ["zone_id", "car_parking_original"],
            },
        ):
            with pyro.poutine.trace():
                base_model(**kwargs)

    new_guide = run_svi_inference(new_model, **minimal_subset)

    minimal_subset_no_outcome = copy.deepcopy(minimal_subset)
    minimal_subset_no_outcome["outcome"] = None

    predictive = PredictiveModel(new_model, guide=new_guide)

    minimal_subset_no_outcome = copy.deepcopy(minimal_subset)
    minimal_subset_no_outcome["outcome"] = None

    with pyro.plate("samples", size=10, dim=-10):
        with pyro.poutine.trace() as tr_with_plate:
            predictive(**minimal_subset_no_outcome)

    with pyro.poutine.trace() as tr:
        predictive(**minimal_subset_no_outcome)

    assert torch.allclose(
        tr_with_plate.trace.nodes["outcome_observed"]["value"]
        .detach()
        .squeeze()[2, 0, :],
        minimal_subset["outcome"],
        atol=7,
    )
    assert torch.allclose(
        tr.trace.nodes["outcome_observed"]["value"].detach().squeeze(),
        minimal_subset["outcome"],
        atol=7,
    )
    assert not torch.allclose(
        tr.trace.nodes["outcome_observed"]["value"].detach().squeeze(),
        minimal_subset["outcome"],
        atol=0.2,
    )
