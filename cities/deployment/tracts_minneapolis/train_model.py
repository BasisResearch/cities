import os

import dill
import pyro
import torch
from torch.utils.data import DataLoader

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

# can be disposed of once you access data in a different manner
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data
from copy import deepcopy
from chirho.observational.handlers import condition

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

# DEBUG

def nonify_dict(d):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = None
        elif isinstance(v, dict):
            d[k] = nonify_dict(v)
    return d

nonified_subset = nonify_dict(deepcopy(subset))
#
# unconditioned_model = TractsModel(
#     **nonified_subset, categorical_levels=ct_dataset_read.categorical_levels
# )
#
# with pyro.poutine.trace() as tr:
#     unconditioned_model(
#         **nonified_subset
#     )
#
# exit()
# # /DEBUG

# We're replacing this with the same thing below, but with an outside-conditioned version.
internally_conditioned_tracts_model = TractsModel(
    **subset, categorical_levels=ct_dataset_read.categorical_levels
)

# DEBUG
unconditioned_model = pyro.poutine.uncondition(internally_conditioned_tracts_model)

with pyro.poutine.trace() as tr:
    unconditioned_model(
        **nonified_subset,
        n=816
    )

SUBSET_SITE_NAME_MAP = {
    "white": "white_original",
    "segregation": "segregation_original",
    "limit": "mean_limit_original",
    "distance": "median_distance",
}

def map_subset_onto_obs(subset):
    obs = dict()

    site_names = ["year", "distance", "white", "segregation", "income", "limit", "median_value", "housing_units"]

    for name in site_names:
        subset_name = SUBSET_SITE_NAME_MAP.get(name, name)
        for k, inner_subset_dict in subset.items():
            if k == "outcome":
                continue
            if subset_name in inner_subset_dict:
                obs[name] = inner_subset_dict[subset_name]
                break

    assert obs.keys() == set(site_names), f"Missing keys: {set(site_names) - obs.keys()}"
    return obs

subset_as_obs = map_subset_onto_obs(subset)

tracts_model = condition(unconditioned_model, data=subset_as_obs)

with pyro.poutine.trace() as tr:
    tracts_model(
        **nonified_subset,
        n=816
    )
# from functools import partial
# tracts_model = partial(
#     unconditioned_model
# )

# import matplotlib.pyplot as plt
# plt.hist(tr.get_trace().nodes["white"]["value"].detach())
# plt.show()

import matplotlib.pyplot as plt
plt.figure()
plt.hist(tr.get_trace().nodes["year"]["value"].detach())
plt.figure()
plt.hist(subset_as_obs["year"].detach())
plt.show()

exit()
# /DEBUG

pyro.clear_param_store()

guide = run_svi_inference(tracts_model, n_steps=2000, lr=0.03, **subset)


##########################################
# save guide and params in the same folder
##########################################


serialized_guide = dill.dumps(guide)
file_path = "tracts_model_guide.pkl"
with open(file_path, "wb") as file:
    file.write(serialized_guide)

param_path = "tracts_model_params.pth"
pyro.get_param_store().save(param_path)
