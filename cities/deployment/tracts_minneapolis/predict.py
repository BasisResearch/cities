import copy
import os

import dill
import pandas as pd
import pyro
import sqlalchemy
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do
from dotenv import load_dotenv
from pyro.infer import Predictive

from cities.modeling.zoning_models.zoning_tracts_sqm_model import (
    TractsModelSqm as TractsModel,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data, select_from_sql

load_dotenv()


DB_USERNAME = os.getenv("DB_USERNAME")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
PASSWORD = os.getenv("PASSWORD")


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
            "housing_units_original",
            "parcel_sqm",
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
            "parcel_sqm",
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
        deploy_path = os.path.join(root, "cities/deployment/tracts_minneapolis")

        guide_path = os.path.join(deploy_path, "tracts_model_guide.pkl")
        self.param_path = os.path.join(deploy_path, "tracts_model_params.pth")

        need_to_train_flag = False
        if not os.path.isfile(guide_path):
            need_to_train_flag = True
            print(f"Warning: '{guide_path}' does not exist.")
        if not os.path.isfile(self.param_path):
            need_to_train_flag = True
            print(f"Warning: '{self.param_path}' does not exist.")

        if need_to_train_flag:
            print("Please run 'train_model.py' to generate the required files.")

        with open(guide_path, "rb") as file:
            guide = dill.load(file)

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

        self.housing_units_std = self.data["continuous"]["housing_units_original"].std()
        self.housing_units_mean = self.data["continuous"][
            "housing_units_original"
        ].mean()

        model = TractsModel(**self.subset, categorical_levels=categorical_levels)
        self.predictive = Predictive(model=model, guide=guide, num_samples=100)

    # these are at the tracts level
    def _tracts_intervention(
        self,
        conn,
        radius_blue,
        limit_blue,
        radius_yellow,
        limit_yellow,
        reform_year,
    ):
        params = {
            "reform_year": reform_year,
            "radius_blue": radius_blue,
            "limit_blue": limit_blue,
            "radius_yellow": radius_yellow,
            "limit_yellow": limit_yellow,
        }
        print(TractsModelPredictor.tracts_intervention_sql % params)
        df = pd.read_sql(
            TractsModelPredictor.tracts_intervention_sql, conn, params=params
        )
        return torch.tensor(df["intervention"].values, dtype=torch.float32)

    def predict_cumulative(self, conn, intervention):
        """Predict the total number of housing units built from 2011-2020 under intervention.

        Returns a dictionary with keys:
        - 'census_tracts': the tracts considered
        - 'housing_units_factual': total housing units built according to real housing data
        - 'housing_units_counterfactual': samples from prediction of total housing units built
        """
        pyro.clear_param_store()
        pyro.get_param_store().load(self.param_path)

        subset_for_preds = copy.deepcopy(self.subset)
        subset_for_preds["continuous"]["housing_units"] = None

        limit_intervention = self._tracts_intervention(conn, **intervention)

        with MultiWorldCounterfactual():
            with do(actions={"limit": limit_intervention}):
                result = self.predictive(**subset_for_preds)["housing_units"].squeeze()[
                    :, 1, :
                ]

        years = self.data["categorical"]["year_original"]
        tracts = self.data["categorical"]["census_tract"]
        f_housing_units = self.data["continuous"]["housing_units_original"]
        cf_housing_units = result * self.housing_units_std + self.housing_units_mean

        # calculate cumulative housing units (factual)
        f_totals = {}
        for i in range(tracts.shape[0]):
            key = tracts[i].item()
            if key not in f_totals:
                f_totals[key] = 0
            f_totals[key] += f_housing_units[i]

        # calculate cumulative housing units (counterfactual)
        cf_totals = {}
        for i in range(tracts.shape[0]):
            year = years[i].item()
            key = tracts[i].item()
            if key not in cf_totals:
                cf_totals[key] = 0
            if year < intervention["reform_year"]:
                cf_totals[key] += f_housing_units[i]
            else:
                cf_totals[key] = cf_totals[key] + cf_housing_units[:, i]
        cf_totals = {k: torch.clamp(v, 0) for k, v in cf_totals.items()}

        census_tracts = list(cf_totals.keys())
        f_housing_units = [f_totals[k] for k in census_tracts]
        cf_housing_units = [cf_totals[k] for k in census_tracts]

        return {
            "census_tracts": census_tracts,
            "housing_units_factual": f_housing_units,
            "housing_units_counterfactual": cf_housing_units,
        }


if __name__ == "__main__":
    import time

    USERNAME = os.getenv("DB_USERNAME")
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    PASSWORD = os.getenv("PASSWORD")

    with sqlalchemy.create_engine(
        f"postgresql://{DB_USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    ).connect() as conn:
        predictor = TractsModelPredictor(conn)
        start = time.time()

        result = predictor.predict_cumulative(
            conn,
            intervention={
                "radius_blue": 300,
                "limit_blue": 0.5,
                "radius_yellow": 700,
                "limit_yellow": 0.7,
                "reform_year": 2015,
            },
        )
        end = time.time()
        print(f"Counterfactual in {end - start} seconds")
