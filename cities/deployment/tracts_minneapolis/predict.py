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

from cities.modeling.zoning_models.zoning_tracts_continuous_interactions_model import (
    TractsModelContinuousInteractions as TractsModel,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data, select_from_sql

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
        when downtown_yn or university_yn then 0
        when year_ < %(reform_year)s then 1
        when distance_to_transit <= %(radius_blue)s then %(limit_blue)s
        when distance_to_transit_line <= %(radius_yellow_line)s
             or distance_to_transit_stop <= %(radius_yellow_stop)s
             then %(limit_yellow)s
        else 1
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

        # R: I assume this this is Jack's workaround to ensure the limits align, correct?
        self.data["continuous"]["mean_limit_original"] = self.obs_limits(conn)

        # R: fix this assertion make sure its satisfied
        # assert (self.data["continuous"]["university_overlap"] > 2).logical_not().all()
        # | (self.data["continuous"]["mean_limit_original"] == 0).all(), \
        # "Mean limit original should be zero wherever university overlap exceeds 2."

        # set to zero whenever the university overlap is above 1
        # # TODO check, this should now be handled at the data processing stage
        # self.data["continuous"]["mean_limit_original"] = torch.where(
        #     self.data["continuous"]["university_overlap"] > 1,
        #     torch.zeros_like(self.data["continuous"]["mean_limit_original"]),
        #     self.data["continuous"]["mean_limit_original"],
        # )

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
        ]

        model = TractsModel(
            **self.subset,
            categorical_levels=categorical_levels,
            housing_units_continuous_interaction_pairs=ins,
        )

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
        """Return the mean parking limits at the tracts level that result from the given intervention.

        Parameters:
        - conn: database connection
        - radius_blue: radius of the blue zone (meters)
        - limit_blue: parking limit for blue zone
        - radius_yellow_line: radius of the yellow zone around lines (meters)
        - radius_yellow_stop: radius of the yellow zone around stops (meters)
        - limit_yellow: parking limit for yellow zone
        - reform_year: year of the intervention

        Returns: Tensor of parking limits sorted by tract and year
        """
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

    def obs_limits(self, conn):
        """Return the observed (factual) parking limits at the tracts level."""
        return self._tracts_intervention(conn, 106.7, 0, 402.3, 804.7, 0.5, 2015)

    def predict_cumulative(self, conn, intervention):
        """Predict the total number of housing units built from 2011-2020 under intervention.

        Returns a dictionary with keys:
        - 'census_tracts': the tracts considered
        - 'housing_units_factual': total housing units built according to real housing data
        - 'housing_units_counterfactual': samples from prediction of total housing units built
        """

        limit_intervention = self._tracts_intervention(conn, **intervention)

        # R: fix this assertion make sure its satisfied
        # assert (self.data["continuous"]["downtown_overlap"] <= 2).all() | (limit_intervention == 0).all(), \
        # "Limit intervention should be zero wherever downtown overlap exceeds 1."

        # R: this shouldn't be required now, remove when confirmed
        # limit_intervention = torch.where(
        #     self.data["continuous"]["university_overlap"] > 2,
        #     torch.zeros_like(limit_intervention),
        #     limit_intervention,
        # )

        # limit_intervention = torch.where(
        #     self.data["continuous"]["downtown_overlap"] > 1,
        #     torch.zeros_like(limit_intervention),
        #     limit_intervention,
        # )

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

        # _____________________________________________
        # R: this is the old code, remove when we reshape the output
        # from above into Michi's desired format
        # presumably outdated

        # with mwc:
        #     result = gather(
        #         result_all, IndexSet(**{"limit": {1}}), event_dims=0
        #     ).squeeze()

        # years = self.data["categorical"]["year_original"]
        # tracts = self.data["categorical"]["census_tract"]
        # f_housing_units = self.data["continuous"]["housing_units_original"]
        # cf_housing_units = result * self.housing_units_std + self.housing_units_mean

        # # Organize cumulative data by year and tract
        # f_data = {}
        # cf_data = {}
        # unique_years = sorted(set(years.tolist()))
        # unique_years = [
        #     year for year in unique_years if year <= 2019
        # ]  # Exclude years after 2019
        # unique_tracts = sorted(set(tracts.tolist()))

        # for year in unique_years:
        #     f_data[year] = {tract: 0 for tract in unique_tracts}
        #     cf_data[year] = {tract: [0] * 100 for tract in unique_tracts}

        # for i in range(tracts.shape[0]):
        #     year = years[i].item()
        #     if year > 2019:
        #         continue  # Skip data for years after 2019
        #     tract = tracts[i].item()

        #     # Update factual data
        #     for y in unique_years:
        #         if y >= year:
        #             f_data[y][tract] += f_housing_units[i].item()

        #     # Update counterfactual data
        #     if year < intervention["reform_year"]:
        #         for y in unique_years:
        #             if y >= year:
        #                 cf_data[y][tract] = [
        #                     x + f_housing_units[i].item() for x in cf_data[y][tract]
        #                 ]
        #     else:
        #         for y in unique_years:
        #             if y >= year:
        #                 cf_data[y][tract] = [
        #                     x + y
        #                     for x, y in zip(
        #                         cf_data[y][tract], cf_housing_units[:, i].tolist()
        #                     )
        #                 ]

        # # Convert to lists for easier JSON serialization
        # housing_units_factual = [
        #     [f_data[year][tract] for tract in unique_tracts] for year in unique_years
        # ]
        # housing_units_counterfactual = [
        #     [cf_data[year][tract] for tract in unique_tracts] for year in unique_years
        # ]

        # ___________________________________________________________
        # TODO remove output not used in debugging, evaluation or on the fronend side
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
            # "years": unique_years,
            # "census_tracts": unique_tracts,
            # "housing_units_factual": housing_units_factual,
            # "housing_units_counterfactual": housing_units_counterfactual,
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
            local_start = time.time()
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
            local_end = time.time()
            print(f"Counterfactual in {local_end - local_start} seconds")
        end = time.time()
        print(f"5 counterfactuals in {end - start} seconds")
