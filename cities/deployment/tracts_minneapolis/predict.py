import copy
import os

import dill
import pandas as pd
import pyro
import torch
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do
from dotenv import load_dotenv
from pyro.infer import Predictive

from cities.modeling.zoning_models.zoning_tracts_population import (
    TractsModelPopulation as TractsModel,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data, select_from_sql

# from cities.modeling.zoning_models.zoning_tracts_sqm_model import (
#     TractsModelSqm as TractsModel,
# )

# from cities.modeling.zoning_models.zoning_tracts_continuous_interactions_model import (
#     TractsModelContinuousInteractions as TractsModel,
# )


load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))


class TractsModelPredictor:
    kwargs = {
        "categorical": ["year", "census_tract", "year_original"],
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
             and (distance_to_transit_line <= %(radius_yellow_line)s
                  or distance_to_transit_stop <= %(radius_yellow_stop)s)
             then %(limit_yellow)s
        when not downtown_yn
             and year_ >= %(reform_year)s
             and distance_to_transit_line > %(radius_yellow_line)s
             and distance_to_transit_stop > %(radius_yellow_stop)s
             then 1
        else limit_con
      end as intervention
    from tracts_model__parcels
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

        self.guide_path = os.path.join(deploy_path, "tracts_model_guide.pkl")
        self.param_path = os.path.join(deploy_path, "tracts_model_params.pth")

        need_to_train_flag = False
        if not os.path.isfile(self.guide_path):
            need_to_train_flag = True
            print(f"Warning: '{self.guide_path}' does not exist.")
        if not os.path.isfile(self.param_path):
            need_to_train_flag = True
            print(f"Warning: '{self.param_path}' does not exist.")

        if need_to_train_flag:
            print("Please run 'train_model.py' to generate the required files.")

        self.data = select_from_sql(
            "select * from tracts_model__census_tracts order by census_tract, year",
            conn,
            TractsModelPredictor.kwargs,
        )

        # set to zero whenever the university overlap is above 1
        # TODO this should be handled at the data processing stage
        self.data["continuous"]["mean_limit_original"] = torch.where(
            self.data["continuous"]["university_overlap"] > 1,
            torch.zeros_like(self.data["continuous"]["mean_limit_original"]),
            self.data["continuous"]["mean_limit_original"],
        )

        self.subset = select_from_data(self.data, TractsModelPredictor.kwargs)

        self.years = self.data["categorical"]["year_original"]
        self.year_ids = self.data["categorical"]["year"]
        self.tracts = self.data["categorical"]["census_tract"]

        categorical_levels = {
            "year": torch.unique(self.subset["categorical"]["year"]),
            "year_original": torch.unique(self.subset["categorical"]["year_original"]),
            "census_tract": torch.unique(self.subset["categorical"]["census_tract"]),
        }

        self.housing_units_std = self.data["continuous"]["housing_units_original"].std()
        self.housing_units_mean = self.data["continuous"][
            "housing_units_original"
        ].mean()

        # interaction_pairs
        # ins = [
        # ("university_overlap", "limit"),
        # ("downtown_overlap", "limit"),
        # ("distance", "downtown_overlap"),
        # ("distance", "university_overlap"),
        # ("distance", "limit"),
        # ("median_value", "segregation"),
        # ("distance", "segregation"),
        # ("limit", "sqm"),
        # ("segregation", "sqm"),
        # ("distance", "white"),
        # ("income", "limit"),
        # ("downtown_overlap", "median_value"),
        # ("downtown_overlap", "segregation"),
        # ("median_value", "white"),
        # ("distance", "income"),
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
            # from density/pop stage 1
            ("population", "sqm"),
            ("density", "income"),
            ("density", "white"),
            ("density", "segregation"),
            ("density", "sqm"),
            ("density", "downtown_overlap"),
            ("density", "university_overlap"),
            ("population", "density"),
        ]

        model = TractsModel(
            **self.subset,
            categorical_levels=categorical_levels,
            housing_units_continuous_interaction_pairs=ins,
        )

        # moved most of this logic here to avoid repeated computations

        with open(self.guide_path, "rb") as file:
            self.guide = dill.load(file)

        pyro.clear_param_store()
        pyro.get_param_store().load(self.param_path)

        self.predictive = Predictive(model=model, guide=self.guide, num_samples=100)

        self.subset_for_preds = copy.deepcopy(self.subset)
        self.subset_for_preds["continuous"]["housing_units"] = None

    # these are at the tracts level
    def _tracts_intervention(
        self,
        conn,
        radius_blue,
        limit_blue,
        radius_yellow_line,
        radius_yellow_stop,
        limit_yellow,
        reform_year,
    ):
        params = {
            "reform_year": reform_year,
            "radius_blue": radius_blue,
            "limit_blue": limit_blue,
            "radius_yellow_line": radius_yellow_line,
            "radius_yellow_stop": radius_yellow_stop,
            "limit_yellow": limit_yellow,
        }
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

        limit_intervention = self._tracts_intervention(conn, **intervention)

        limit_intervention = torch.where(
            self.data["continuous"]["university_overlap"] > 2,
            torch.zeros_like(limit_intervention),
            limit_intervention,
        )

        limit_intervention = torch.where(
            self.data["continuous"]["downtown_overlap"] > 1,
            torch.zeros_like(limit_intervention),
            limit_intervention,
        )

        with MultiWorldCounterfactual() as mwc:
            with do(actions={"limit": limit_intervention}):
                result_all = self.predictive(**self.subset_for_preds)["housing_units"]
        with mwc:
            result_f = gather(
                result_all, IndexSet(**{"limit": {0}}), event_dims=0
            ).squeeze()
            result_cf = gather(
                result_all, IndexSet(**{"limit": {1}}), event_dims=0
            ).squeeze()

        obs_housing_units_raw = self.data["continuous"]["housing_units_original"]
        f_housing_units_raw = (
            result_f * self.housing_units_std + self.housing_units_mean
        ).clamp(min=0)
        cf_housing_units_raw = (
            result_cf * self.housing_units_std + self.housing_units_mean
        ).clamp(min=0)

        # calculate cumulative housing units (factual)
        obs_limits = {}
        cf_limits = {}
        obs_cumsums = {}
        f_cumsums = {}
        cf_cumsums = {}
        for key in self.tracts.unique():
            obs_units = []
            f_units = []
            cf_units = []
            obs_limits_list = []
            cf_limits_list = []
            for year in self.years.unique():

                mask = (self.tracts == key) & (self.years == year)

                obs_units.append(obs_housing_units_raw[mask])

                obs_limits_list.append(
                    self.data["continuous"]["mean_limit_original"][mask]
                )
                cf_limits_list.append(limit_intervention[mask])

                f_units.append(f_housing_units_raw[:, mask])
                cf_units.append(cf_housing_units_raw[:, mask])

            obs_cumsum = torch.cumsum(torch.stack(obs_units), dim=0).flatten()
            obs_limits[key] = torch.stack(obs_limits_list).flatten()
            cf_limits[key] = torch.stack(cf_limits_list).flatten()
            f_cumsum = torch.cumsum(torch.stack(f_units), dim=0).squeeze()
            cf_cumsum = torch.cumsum(torch.stack(cf_units), dim=0).squeeze()

            obs_cumsums[key] = obs_cumsum
            f_cumsums[key] = f_cumsum
            cf_cumsums[key] = cf_cumsum

        # presumably outdated

        tracts = self.data["categorical"]["census_tract"]

        # calculate cumulative housing units (factual)
        f_totals = {}
        for i in range(tracts.shape[0]):
            key = tracts[i].item()
            if key not in f_totals:
                f_totals[key] = 0
            f_totals[key] += obs_housing_units_raw[i]

        # calculate cumulative housing units (counterfactual)
        cf_totals = {}
        for i in range(tracts.shape[0]):
            year = self.years[i].item()
            key = tracts[i].item()
            if key not in cf_totals:
                cf_totals[key] = 0
            if year < intervention["reform_year"]:
                cf_totals[key] += obs_housing_units_raw[i]
            else:
                cf_totals[key] = cf_totals[key] + cf_housing_units_raw[:, i]
        cf_totals = {k: torch.clamp(v, 0) for k, v in cf_totals.items()}

        census_tracts = list(cf_totals.keys())
        f_housing_units = [f_totals[k] for k in census_tracts]
        cf_housing_units = [cf_totals[k] for k in census_tracts]

        return {
            "obs_cumsums": obs_cumsums,
            "f_cumsums": f_cumsums,
            "cf_cumsums": cf_cumsums,
            "limit_intervention": limit_intervention,
            "obs_limits": obs_limits,
            "cf_limits": cf_limits,
            "raw_obs_housing_units": obs_housing_units_raw,
            "raw_f_housing_units": f_housing_units_raw,
            "raw_cf_housing_units": cf_housing_units_raw,
            # presumably outdated
            "census_tracts": census_tracts,
            "housing_units_factual": f_housing_units,
            "housing_units_counterfactual": cf_housing_units,
        }

        # return {
        #     "census_tracts": census_tracts,
        #     "housing_units_factual": f_housing_units,
        #     "housing_units_counterfactual": cf_housing_units,
        #     "limit_intervention": limit_intervention,
        # }


if __name__ == "__main__":
    import time

    from cities.utils.data_loader import db_connection

    with db_connection() as conn:
        predictor = TractsModelPredictor(conn)
        start = time.time()

        for iter in range(5):
            result = predictor.predict_cumulative(
                conn,
                intervention={
                    "radius_blue": 350,
                    "limit_blue": 0,
                    "radius_yellow_line": 1320,
                    "radius_yellow_stop": 2640,
                    "limit_yellow": 0.5,
                    "reform_year": 2015,
                },
            )
        end = time.time()
        print(f"Counterfactual in {end - start} seconds")
