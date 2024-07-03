import os

import copy
import torch
from torch.utils.data import DataLoader

import pyro
from cities.modeling.add_causal_layer import AddCausalLayer
from cities.modeling.simple_linear import SimpleLinear
from cities.modeling.svi_inference import run_svi_inference
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

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
    "continuous": {"x_con": x_con, "y_mixed": y_mixed   },
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
            base_model_synthetic(**synthetic_minimal_outcome)

    new_guide_synthetic = run_svi_inference(
        new_model_synthetic, **synthetic_minimal_outcome, vi_family=AutoDiagonalNormal
    )
    # note AutoMultivariateNormal fails, not clear why

    predictive_synthetic = Predictive(
    new_model_synthetic, guide=new_guide_synthetic, num_samples=500, parallel=True
        )

    synthetic_minimal = copy.deepcopy(synthetic_minimal_outcome)
    synthetic_minimal['outcome'] = None

    samples_synthetic = predictive_synthetic(**synthetic_minimal)

    w_x_cat_to_x_con = samples_synthetic['weights_categorical_x_cat_x_con'].squeeze()

    cat0_mean = w_x_cat_to_x_con[:, 0].mean()
    cat1_mean = w_x_cat_to_x_con[:, 1].mean()
    cat2_mean = w_x_cat_to_x_con[:, 2].mean()

    assert torch.allclose(torch.tensor([cat0_mean, cat1_mean, cat2_mean]),
                     torch.tensor([1.0, 2.0, 3.0]), atol=0.2)


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
                base_model(**minimal_subset)

    new_guide = run_svi_inference(new_model, **minimal_subset)

    predictive = Predictive(new_model, guide=new_guide, num_samples=20, parallel=True)

    samples = predictive(**minimal_subset)

    assert samples["weights_categorical_zone_id_parcel_area"].squeeze().shape == (20, 4)
