import os

import pandas as pd
import torch
from dotenv import load_dotenv

from cities.modeling.zoning_models.ts_model_components import (
    reshape_into_time_series,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_sql

# TODO load the right model
# from cities.modeling.zoning_models.zoning_tracts_continuous_interactions_model import (
#    TractsModelContinuousInteractions as TractsModel,
# )


load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))


num_samples = 100

# this disables assertions for speed
dev_mode = False


class TractsModelPredictor:
    kwargs = {
        "categorical": ["year", "census_tract", "year_original"],
        "continuous": {
            "housing_units",
            "housing_units_original",
            "total_value",
            "total_value_original",
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
            "downtown_overlap_original",
            "university_overlap",
            "university_overlap_original",
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

        # --------------------------
        # data loading and processing
        # --------------------------

        data = select_from_sql(
            "select * from tracts_model__census_tracts order by census_tract, year",
            conn,
            TractsModelPredictor.kwargs,
        )

        # time series modeling assumes dim -1 is time, dim -2 is series
        # reshaping data to fit this assumption
        # potentially this can migrate to SQL

        unique_series = reshape_into_time_series(
            data["continuous"]["housing_units"],
            series_idx=data["categorical"]["census_tract"],
            time_idx=data["categorical"]["year"],
        )["unique_series"]

        data["reshaped"] = {}
        data["reshaped"]["continuous"] = {}
        data["reshaped"]["categorical"] = {}

        for key, val in data["continuous"].items():
            data["reshaped"]["continuous"][key] = reshape_into_time_series(
                val,
                series_idx=data["categorical"]["census_tract"],
                time_idx=data["categorical"]["year"],
            )["reshaped_variable"]

        for key, val in data["categorical"].items():
            data["reshaped"]["categorical"][key] = reshape_into_time_series(
                val,
                series_idx=data["categorical"]["census_tract"],
                time_idx=data["categorical"]["year"],
            )["reshaped_variable"]

        data["init_state"] = data["reshaped"]["continuous"]["housing_units"][
            ..., 0
        ].unsqueeze(-1)
        data["init_idx"] = data["reshaped"]["categorical"]["census_tract"][
            ..., 0
        ].unsqueeze(-1)

        data["housing_units_mean"] = data["reshaped"]["continuous"][
            "housing_units_original"
        ].mean()
        data["housing_units_std"] = data["reshaped"]["continuous"][
            "housing_units_original"
        ].std()

        self.data = data
        # TODO model will be revised
        # --------------------------
        # model loading
        # --------------------------

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

        # --------------------------
        # intervention helpers
        # --------------------------

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
