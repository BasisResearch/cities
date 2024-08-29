import copy
import time

import dill
import pyro
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual

# import chirho
from chirho.interventional.handlers import do
from pyro.infer import Predictive

from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

# can be disposed of once you access data in a different manner
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import load_sql_df, select_from_sql

root = find_repo_root()


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

subset = select_from_sql("select * from dev.tracts_model__census_tracts", kwargs)

categorical_levels = {
    "year": torch.unique(subset["categorical"]["year"]),
    "census_tract": torch.unique(subset["categorical"]["census_tract"]),
}

subset_for_preds = copy.deepcopy(subset)
subset_for_preds["continuous"]["housing_units"] = None


########################
# load trained model (run `train_model.py` first)
########################

tracts_model = TractsModel(**subset, categorical_levels=categorical_levels)

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

parcel_intervention_sql = """
select
  census_tract,
  year_,
  case
    when downtown_yn then 0
    when not downtown_yn and year_ >= %(reform_year)s and distance_to_transit <= %(radius_blue)s then %(limit_blue)s
    when not downtown_yn and year_ >= %(reform_year)s and distance_to_transit > %(radius_blue)s and distance_to_transit <= %(radius_yellow)s then %(limit_yellow)s
    when not downtown_yn and year_ > %(reform_year)s and distance_to_transit > %(radius_yellow)s then 1
    else limit_con
  end as intervention
from dev.tracts_model__parcels
"""


# these are at the parcel level
def values_intervention(
    radius_blue, limit_blue, radius_yellow, limit_yellow, reform_year=2015
):
    params = {
        "reform_year": reform_year,
        "radius_blue": radius_blue,
        "limit_blue": limit_blue,
        "radius_yellow": radius_yellow,
        "limit_yellow": limit_yellow,
    }
    return load_sql_df(parcel_intervention_sql, params)


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
    tracts_intervention_sql = f"""
    with parcel_interventions as ({parcel_intervention_sql})
    select
        census_tract,
        year_,
        avg(intervention) as intervention
    from parcel_interventions
    group by census_tract, year_
    order by census_tract, year_
    """
    df = load_sql_df(
        tracts_intervention_sql,
        {
            "reform_year": reform_year,
            "radius_blue": radius_blue,
            "limit_blue": limit_blue,
            "radius_yellow": radius_yellow,
            "limit_yellow": limit_yellow,
        },
    )
    return torch.tensor(df["intervention"].values, dtype=torch.float32)


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
    with do(actions={"limit": t_intervention}):
        samples = predictive(**subset_for_preds)


assert samples["limit"].shape[:-1] == torch.Size([100, 2, 1, 1, 1])
