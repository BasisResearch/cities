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

# can be disposed of once you access data in a different manner
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

root = find_repo_root()


#####################
# data load and prep
#####################

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

subset_for_preds = copy.deepcopy(subset)
subset_for_preds["continuous"]["housing_units"] = None


########################
# load trained model (run `train_model.py` first)
########################

tracts_model = TractsModel(
    **subset, categorical_levels=ct_dataset_read.categorical_levels
)

pyro.clear_param_store()

guide_path = "tracts_model_guide.pkl"
param_path = "tracts_model_params.pth"

with open(guide_path, "rb") as file:
    guide = dill.load(file)

pyro.get_param_store().load(param_path)

predictive = Predictive(
    model=tracts_model,
    guide=guide,
    num_samples=100,
)


############################################################
# define interventions parametrized as in the intended query
############################################################


# these are at the parcel level
def values_intervention(
    radius_blue, limit_blue, radius_yellow, limit_yellow, reform_year=2015
):

    # don't want to load large data multiple times
    # note we'll need to generate these datasets anew once we switch to the new data pipeline

    if not hasattr(values_intervention, "global_census_ids"):
        values_intervention.global_census_ids = pd.read_csv(
            os.path.join(root, "data/minneapolis/processed/census_ids.csv")
        )

        values_intervention.global_data = pd.read_csv(
            os.path.join(
                root,
                "data/minneapolis/processed/census_tract_intervention_required.csv",
            )
        )

        data = values_intervention.global_data
        census_ids = values_intervention.global_census_ids
        values_intervention.global_data = data[
            (data["census_tract"].isin(census_ids["census_tract"]))
            & (data["year"].isin(census_ids["year"]))
        ]

    data = values_intervention.global_data.copy()

    intervention = copy.deepcopy(values_intervention.global_data["limit_con"])
    downtown = data["downtown_yn"]
    new_blue = (
        (~downtown)
        & (data["year"] >= reform_year)
        & (data["distance_to_transit"] <= radius_blue)
    )
    new_yellow = (
        (~downtown)
        & (data["year"] >= reform_year)
        & (data["distance_to_transit"] > radius_blue)
        & (data["distance_to_transit"] <= radius_yellow)
    )
    new_other = (
        (~downtown)
        & (data["year"] > reform_year)
        & (data["distance_to_transit"] > radius_yellow)
    )

    intervention[downtown] = 0.0
    intervention[new_blue] = limit_blue
    intervention[new_yellow] = limit_yellow
    intervention[new_other] = 1.0

    data["intervention"] = intervention

    return data


# generate three interventions at the parcel level

start = time.time()
simple_intervention = values_intervention(300, 0.5, 700, 0.7, reform_year=2015)
end = time.time()
print("Time to run values_intervention 1: ", end - start)
start2 = time.time()
simple_intervention2 = values_intervention(400, 0.5, 800, 0.6, reform_year=2013)
end2 = time.time()
print("Time to run values_intervention 2: ", end2 - start2)
start3 = time.time()
simple_intervention3 = values_intervention(200, 0.4, 1000, 0.65, reform_year=2013)
end3 = time.time()
print("Time to run values_intervention 3: ", end3 - start3)


# these are at the tracts level


def tracts_intervention(
    radius_blue, limit_blue, radius_yellow, limit_yellow, reform_year=2015
):

    parcel_interventions = values_intervention(
        radius_blue, limit_blue, radius_yellow, limit_yellow, reform_year=reform_year
    )

    aggregate = (
        parcel_interventions[["census_tract", "year", "intervention"]]
        .groupby(["census_tract", "year"])
        .mean()
        .reset_index()
    )

    if not hasattr(tracts_intervention, "global_census_ids"):

        tracts_intervention.global_valid_pairs = set(
            zip(
                values_intervention.global_census_ids["census_tract"],
                values_intervention.global_census_ids["year"],
            )
        )

    subaggregate = aggregate[
        aggregate[["census_tract", "year"]]
        .apply(tuple, axis=1)
        .isin(tracts_intervention.global_valid_pairs)
    ].copy()

    return torch.tensor(list(subaggregate["intervention"]))


# generate two interventions at the tracts level

start = time.time()
t_intervention = tracts_intervention(300, 0.5, 700, 0.7, reform_year=2015)
end = time.time()
print("Time to run tracts_intervention 1: ", end - start)

start2 = time.time()
t_intervention2 = tracts_intervention(400, 0.5, 800, 0.6, reform_year=2013)
end2 = time.time()
print("Time to run tracts_intervention 2: ", end2 - start2)


##################################
# use interventions with the model
##################################

with MultiWorldCounterfactual() as mwc:
    with do(actions={"limit": torch.tensor(0.0)}):
        samples = predictive(**subset_for_preds)


assert samples["limit"].shape == torch.Size([100, 2, 1, 1, 1, 816])
