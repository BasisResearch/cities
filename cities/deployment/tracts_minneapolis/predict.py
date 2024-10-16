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

num_samples = 100


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

        self.predictive = Predictive(
            model=model, guide=self.guide, num_samples=num_samples
        )

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

            key_str = str(key.item())
            obs_cumsum = torch.cumsum(torch.stack(obs_units), dim=0).flatten()
            obs_limits[key_str] = torch.stack(obs_limits_list).flatten()
            cf_limits[key_str] = torch.stack(cf_limits_list).flatten()
            f_cumsum = torch.cumsum(torch.stack(f_units), dim=0).squeeze()
            cf_cumsum = torch.cumsum(torch.stack(cf_units), dim=0).squeeze()

            obs_cumsums[key_str] = obs_cumsum
            f_cumsums[key_str] = f_cumsum
            cf_cumsums[key_str] = cf_cumsum

        assert list(obs_cumsums.keys()) == [
            str(_) for _ in self.tracts.unique().tolist()
        ]

        # R: I'd recommend keeping "cumsums", as well as "observed/factual/counterfactual"
        # in variable names
        # to make terminology clear and transparent
        cumsums_observed = torch.stack(list(obs_cumsums.values())).T.tolist()

        cumsums_factual = [
            [_.tolist() for _ in __.unbind(dim=-2)]
            for __ in torch.stack(list(f_cumsums.values())).unbind(dim=-2)
        ]

        cumsums_counterfactual = [
            [_.tolist() for _ in __.unbind(dim=-2)]
            for __ in torch.stack(list(cf_cumsums.values())).unbind(dim=-2)
        ]

        assert (
            len(cumsums_factual)
            == len(cumsums_observed)
            == len(cumsums_counterfactual)
            == 10
        )
        #  the number of years
        assert (
            len(cumsums_factual[0]) == len(cumsums_counterfactual[0]) == 113
        )  # the number of unique tracts
        assert (
            len(cumsums_factual[0][0])
            == len(cumsums_counterfactual[0][0])
            == num_samples
        )

        return {
            # these are lists whose structures are dictated
            # by the frontend demands
            "census_tracts": list(obs_cumsums.keys()),
            "years": self.years.unique().tolist(),
            "cumsums_observed": cumsums_observed,
            "cumsums_factual": cumsums_factual,
            "cumsums_counterfactual": cumsums_counterfactual,
            # more direct dictionaries used for notebooks and debugging
            # if they slow anything down
            # we can revisit and make an optional output
            "obs_cumsums": obs_cumsums,
            "f_cumsums": f_cumsums,
            "cf_cumsums": cf_cumsums,
            "limit_intervention": limit_intervention,
            "obs_limits": obs_limits,
            "cf_limits": cf_limits,
            "raw_obs_housing_units": obs_housing_units_raw,
            "raw_f_housing_units": f_housing_units_raw,
            "raw_cf_housing_units": cf_housing_units_raw,
        }


# This the desired structure of the output
# (except, we need to correct for the observed/factual distinction
# (and make our terminology consistent with the concepts)
# {
#     "census_tracts": ["27053000100", "27053000200", ...],  # List of census tract IDs
#     "years": [2011, 2012, 2013, ..., 2019],  # List of years

#     "housing_units_factual": [
#         [100, 150, ...],  # Cumulative counts for each tract in 2011
#         [120, 180, ...],  # Cumulative counts for each tract in 2012
#         ...
#     ],

#     "housing_units_counterfactual": [
#         [  # Year 2011
#             [101, 102, ..., 105],  # 100 samples for tract 27053000100
#             [151, 153, ..., 158],  # 100 samples for tract 27053000200
#             ...
#         ],
#         [  # Year 2012
#             [122, 124, ..., 128],  # 100 samples for tract 27053000100
#             [182, 185, ..., 190],  # 100 samples for tract 27053000200
#             ...
#         ],
#         ...
#     ]
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
