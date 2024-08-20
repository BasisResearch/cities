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

tracts_model = TractsModel(
    **subset, categorical_levels=ct_dataset_read.categorical_levels
)

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
