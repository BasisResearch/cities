import os
import time

import dill
import pyro
import torch
from dotenv import load_dotenv

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_population import (
    TractsModelPopulation as TractsModel,
)

# from cities.modeling.zoning_models.zoning_tracts_continuous_interactions_model import (
#    # TractsModelContinuousInteractions as TractsModel,
# )
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import db_connection, select_from_sql

# from cities.modeling.zoning_models.zoning_tracts_model import TractsModel
# from cities.modeling.zoning_models.zoning_tracts_sqm_model import (
#     TractsModelSqm as TractsModel,
# )


n_steps = 1500

load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))

#####################
# data load and prep
#####################

kwargs = {
    "categorical": [
        "year",
        "census_tract",
    ],
    "continuous": {
        "housing_units",
        "housing_units_original",
        "total_value",
        "total_population",
        "population_density",
        "median_value",
        "mean_limit_original",
        "median_distance",
        "income",
        "segregation_original",
        "white_original",
        "parcel_sqm",
        "downtown_overlap",
        "university_overlap",
    },
    "outcome": "housing_units",
}

load_start = time.time()
with db_connection() as conn:
    subset = select_from_sql(
        "select * from dev.tracts_model__census_tracts order by census_tract, year",
        conn,
        kwargs,
    )
load_end = time.time()
print(f"Data loaded in {load_end - load_start} seconds")

#############################
# instantiate and train model
#############################

# interaction terms
# ins = [
#     ("university_overlap", "limit"),
#     ("downtown_overlap", "limit"),
#     ("distance", "downtown_overlap"),
#     ("distance", "university_overlap"),
#     ("distance", "limit"),
#     ("median_value", "segregation"),
#     ("distance", "segregation"),
#     ("limit", "sqm"),
#     ("segregation", "sqm"),
#     ("distance", "white"),
#     ("income", "limit"),
#     ("downtown_overlap", "median_value"),
#     ("downtown_overlap", "segregation"),
#     ("median_value", "white"),
#     ("distance", "income"),
# ]


ins = [
    ("university_overlap", "limit"),
    ("downtown_overlap", "limit"),
    ("distance", "downtown_overlap"),
    ("distance", "university_overlap"),
    ("distance", "limit"),
    ("median_value", "segregation"),
    ("distance", "segregation"),
    ("limit", "sqm"),
    ("segregation", "sqm"),
    ("distance", "white"),
    ("income", "limit"),
    ("downtown_overlap", "median_value"),
    ("downtown_overlap", "segregation"),
    ("median_value", "white"),
    ("distance", "income"),
    ("population", "sqm"),
    ("density", "income"),
    ("density", "white"),
    ("density", "segregation"),
    ("density", "sqm"),
    ("density", "downtown_overlap"),
    ("density", "university_overlap"),
    ("population", "density"),
]


# model
tracts_model = TractsModel(
    **subset,
    categorical_levels={
        "year": torch.unique(subset["categorical"]["year"]),
        "census_tract": torch.unique(subset["categorical"]["census_tract"]),
    },
    housing_units_continuous_interaction_pairs=ins,
)

pyro.clear_param_store()

guide = run_svi_inference(tracts_model, n_steps=n_steps, lr=0.03, plot=False, **subset)

##########################################
# save guide and params in the same folder
##########################################
root = find_repo_root()

deploy_path = os.path.join(root, "cities/deployment/tracts_minneapolis")
guide_path = os.path.join(deploy_path, "tracts_model_guide.pkl")
param_path = os.path.join(deploy_path, "tracts_model_params.pth")

serialized_guide = dill.dumps(guide)
with open(guide_path, "wb") as file:
    file.write(serialized_guide)

with open(param_path, "wb") as file:
    pyro.get_param_store().save(param_path)
