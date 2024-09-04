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

#############################
# instantiate and train model
#############################

nonified_subset = nonify_dict(subset)

tracts_model = TractsModel(
    **subset,  # pass the data here so that categorical levels are properly constructed.
    categorical_levels=ct_dataset_read.categorical_levels
)

# We can partially evaluate with the nonified subset to get an unconditioned model.
unconditioned_model = partial(tracts_model, n=816, **nonified_subset)

# DEBUG

# Tracing the unconditioned forward
with pyro.poutine.trace() as prior_trace:
    unconditioned_model()


subset_as_obs = map_subset_onto_obs(
    subset,
    site_names=["year", "distance", "white", "segregation", "income", "limit", "median_value", "housing_units"]
)

conditioned_model = condition(unconditioned_model, data=subset_as_obs)

# with pyro.poutine.trace() as conditioned_trace:
#     conditioned_model()
#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(prior_trace.get_trace().nodes["year"]["value"].detach())
# plt.suptitle("Prior")
#
# plt.figure()
# plt.hist(subset_as_obs["year"].detach())
# plt.suptitle("Data")
#
# plt.figure()
# plt.hist(conditioned_trace.get_trace().nodes["year"]["value"].detach())
# plt.suptitle("Conditioned")
#
# plt.show()
#
# exit()
# /DEBUG

pyro.clear_param_store()

guide = run_svi_inference(conditioned_model, n_steps=5000, lr=0.005, vi_family=AutoDelta)


##########################################
# save guide and params in the same folder
##########################################


serialized_guide = dill.dumps(guide)
file_path = "tracts_model_guide.pkl"
with open(file_path, "wb") as file:
    file.write(serialized_guide)

param_path = "tracts_model_params.pth"
pyro.get_param_store().save(param_path)
