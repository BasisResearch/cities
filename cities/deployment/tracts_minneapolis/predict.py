import copy
import os

import dill
import pandas as pd
import pyro
from pyro.infer import Predictive
import sqlalchemy
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.interventional.handlers import do

from cities.modeling.zoning_models.zoning_tracts_model import TractsModel

from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_sql, select_from_data


class TractsModelPredictor:
    kwargs = {
        "categorical": ["year", "year_original", "census_tract"],
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

    kwargs_subset = {
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

    parcel_intervention_sql = """
    select
      census_tract,
      year_,
      case
        when downtown_yn then 0
        when not downtown_yn
             and year_ >= %(reform_year)s
             and distance_to_transit <= %(radius_blue)s
             then %(limit_blue)s
        when not downtown_yn
             and year_ >= %(reform_year)s
             and distance_to_transit > %(radius_blue)s
             and distance_to_transit <= %(radius_yellow)s
             then %(limit_yellow)s
        when not downtown_yn
             and year_ >= %(reform_year)s
             and distance_to_transit > %(radius_yellow)s
             then 1
        else limit_con
      end as intervention
    from dev.tracts_model__parcels
    """

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

    def __init__(self, conn):
        self.conn = conn

        root = find_repo_root()

        guide_path = os.path.join(root, "tracts_model_guide.pkl")
        with open(guide_path, "rb") as file:
            guide = dill.load(file)

        self.param_path = os.path.join(root, "tracts_model_params.pth")

        self.data = select_from_sql(
            "select * from dev.tracts_model__census_tracts",
            conn,
            TractsModelPredictor.kwargs,
        )
        self.subset = select_from_data(self.data, TractsModelPredictor.kwargs_subset)

        categorical_levels = {
            "year": torch.unique(self.subset["categorical"]["year"]),
            "census_tract": torch.unique(self.subset["categorical"]["census_tract"]),
        }

        model = TractsModel(**self.subset, categorical_levels=categorical_levels)
        self.predictive = Predictive(model=model, guide=guide, num_samples=100)

    # these are at the tracts level
    def _tracts_intervention(
        self, radius_blue, limit_blue, radius_yellow, limit_yellow, reform_year=2015
    ):
        params = {
            "reform_year": reform_year,
            "radius_blue": radius_blue,
            "limit_blue": limit_blue,
            "radius_yellow": radius_yellow,
            "limit_yellow": limit_yellow,
        }
        df = pd.read_sql(
            TractsModelPredictor.tracts_intervention_sql, self.conn, params=params
        )
        return torch.tensor(df["intervention"].values, dtype=torch.float32)

    def predict(self, intervention=None, samples=100):
        pyro.clear_param_store()
        pyro.get_param_store().load(self.param_path)

        subset_for_preds = copy.deepcopy(self.subset)
        subset_for_preds["continuous"]["housing_units"] = None

        if intervention is None:
            result = self.predictive(**subset_for_preds)["housing_units"]
        else:
            intervention = self._tracts_intervention(**intervention)
            print(intervention.shape, intervention)
            with MultiWorldCounterfactual():
                with do(actions={"limit": intervention}):
                    result = self.predictive(**subset_for_preds)["housing_units"]

        self.data["categorical"]["year_original"], self.data["categorical"][
            "census_tract"
        ], result
        return result


if __name__ == "__main__":
    import time

    USERNAME = os.getenv("USERNAME")
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")

    with sqlalchemy.create_engine(
        f"postgresql://{USERNAME}@{HOST}/{DATABASE}"
    ).connect() as conn:
        predictor = TractsModelPredictor(conn)

        start = time.time()
        print(predictor.predict().shape)
        end = time.time()
        print(f"Predicted in {end - start} seconds")

        start = time.time()
        print(
            predictor.predict(
                intervention={
                    "radius_blue": 300,
                    "limit_blue": 0.5,
                    "radius_yellow": 700,
                    "limit_yellow": 0.7,
                }
            ).shape
        )
        end = time.time()
        print(f"Counterfactual in {end - start} seconds")
