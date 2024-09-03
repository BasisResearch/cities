import dill
import os
import pyro
import torch
import time
import sqlalchemy

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel
from cities.utils.data_loader import select_from_sql

USERNAME = os.getenv("USERNAME")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")

#####################
# data load and prep
#####################

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

load_start = time.time()
with sqlalchemy.create_engine(
    f"postgresql://{USERNAME}@{HOST}/{DATABASE}"
).connect() as conn:
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

tracts_model = TractsModel(
    **subset,
    categorical_levels={
        "year": torch.unique(subset["categorical"]["year"]),
        "census_tract": torch.unique(subset["categorical"]["census_tract"]),
    },
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
