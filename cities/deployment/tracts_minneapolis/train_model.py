import dill
import pyro
import torch

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

from cities.utils.data_loader import select_from_sql


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

subset = select_from_sql(
    "select * from dev.tracts_model__census_tracts order by census_tract, year", kwargs
)

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
