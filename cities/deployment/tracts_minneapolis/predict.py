import os

import dill
import pandas as pd
import pyro
import torch
from chirho.interventional.handlers import do
from dotenv import load_dotenv

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.ts_model_components import prepare_zoning_data_for_ts
from cities.modeling.zoning_models.zoning_tracts_ts_model import (
    TractsModelCumulativeAR1 as TractsModel,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_sql
from cities.utils.plot_ts import summarize_time_series

load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))


num_samples = 200
num_steps = 3000

# this disables assertions for speed
# running as main in dev mode will enforce model re-training
dev_mode = True


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

        self.num_steps = num_steps
        self.num_samples = num_samples

        # --------------------------
        # data loading and processing
        # --------------------------

        data = select_from_sql(
            "select * from tracts_model__census_tracts order by census_tract, year",
            conn,
            TractsModelPredictor.kwargs,
        )

        data["continuous"]["mean_limit_original"] = self.obs_limits(conn)

        # time series modeling assumes dim -1 is time, dim -2 is series
        # reshaping data to fit this assumption
        # potentially this can migrate to SQL

        data, nonified_data = prepare_zoning_data_for_ts(data)

        self.data = data
        self.nonified_data = nonified_data

        self.categorical_levels = {
            "year": torch.unique(self.data["categorical"]["year"]),
            "year_original": torch.unique(self.data["categorical"]["year_original"]),
            "census_tract": torch.unique(self.data["categorical"]["census_tract"]),
        }

        self.census_tracts = self.data["init_idx"].squeeze().numpy().tolist()
        self.years = (
            self.data["categorical"]["year_original"].squeeze().numpy().tolist()
        )

        # TODO model will be revised
        # --------------------------
        # model loading
        # --------------------------

        # interaction pairs
        self.ins = [
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

        self.model = TractsModel(
            self.data,
            housing_units_continuous_parents_names=[
                "median_value",
                "distance",
                "income",
                "white",
                "limit",
                "segregation",
                "sqm",
                "downtown_overlap",
                "university_overlap",
            ],
            housing_units_continuous_interaction_pairs=self.ins,
            leeway=0.9,
        )

        self.root = find_repo_root()

        self.deploy_path = os.path.join(
            self.root, "cities/deployment/tracts_minneapolis"
        )
        self.guide_path = os.path.join(self.deploy_path, "tracts_model_guide.pkl")
        self.param_path = os.path.join(self.deploy_path, "tracts_model_params.pth")

        need_to_train_flag = False
        if not os.path.isfile(self.guide_path):
            need_to_train_flag = True
            print(f"Warning: '{self.guide_path}' does not exist.")
        if not os.path.isfile(self.param_path):
            need_to_train_flag = True
            print(f"Warning: '{self.param_path}' does not exist.")

        if need_to_train_flag:

            if dev_mode:
                print("Running in dev mode. Training the model.")
                self.train_model(override=True)

            else:
                raise Exception(
                    "Please run the 'train_model()' method or instantiate the predictor in dev mode to train the model."
                )

        else:
            pyro.clear_param_store()

            with open(self.guide_path, "rb") as file:
                self.guide = dill.load(file)

            pyro.get_param_store().load(self.param_path)

            self.predictive = pyro.infer.Predictive(
                self.model, guide=self.guide, num_samples=self.num_samples
            )

        # add observed values, transform into list for output
        self.observed_housing_cumulative = self.data["reshaped"]["continuous"][
            "housing_units_cumulative_original"
        ]

        observed_tensor = self.observed_housing_cumulative

        self.observed_housing_cumulative_list = [
            observed_tensor[:, year].tolist() for year in range(10)
        ]

        for year in range(10):
            for series in range(113):
                assert (
                    self.observed_housing_cumulative_list[year][series]
                    == observed_tensor[series, year].item()
                )

        # factual predictions don't depend on the intervention
        # no need to compute them multiple times
        if self.predictive is not None:
            self.clear_reshaped_model_data
            self.factual_samples = self.predictive(data=self.nonified_data)

            self.factual_samples["destandardized_housing_units_cumulative"] = (
                self.factual_samples["predicted_housing_units_cumulative"]
                * self.data["housing_units_cumulative_std"]
                + self.data["housing_units_cumulative_mean"]
            )

            self.factual_summary = summarize_time_series(
                self.factual_samples,
                self.observed_housing_cumulative,
                y_site="destandardized_housing_units_cumulative",
                clamp_at_zero=True,
                compute_metrics=False,
            )

            self.factual_means_list = self.convert_dict_to_list(
                self.factual_summary["series_mean_pred"]
            )
            self.factual_low_list = self.convert_dict_to_list(
                self.factual_summary["series_low_pred"]
            )
            self.factual_high_list = self.convert_dict_to_list(
                self.factual_summary["series_high_pred"]
            )
            self.factual_samples_list = self.generate_samples_list(
                self.factual_samples["destandardized_housing_units_cumulative"]
            )

    @staticmethod
    def convert_dict_to_list(data_dict):
        result = []
        for year in range(10):
            year_list = [data_dict[series][year].item() for series in data_dict.keys()]
            result.append(year_list)

        for year in range(10):
            for series in data_dict.keys():
                assert data_dict[series][year].item() == result[year][series]

        return result

    @staticmethod
    def generate_samples_list(factual_samples, num_years=10, num_series=113):

        samples_list = []
        for year in range(num_years):
            samples_year_list = []
            samples_year = factual_samples[..., year].clamp(min=0)
            for series in range(num_series):
                samples_series = samples_year[..., series].flatten().tolist()
                samples_year_list.append(samples_series)
            samples_list.append(samples_year_list)

        for year in range(num_years):
            for series in range(num_series):
                for sample in [0, 15, 60, 120, 190]:
                    assert (
                        factual_samples.clamp(min=0)[sample, 0, 0, series, year].item()
                        == samples_list[year][series][sample]
                    ), (
                        f"Assertion failed for year={year}, series={series}, sample={sample}. "
                        f"Expected {factual_samples.clamp(min=0)[sample, 0, 0, series, year].item()}, "
                        f"got {samples_list[year][series][sample]}"
                    )

        return samples_list

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

    # for speedup, ts-reshaped predictors are chached in inference
    # but they should be cleared before prediction tasks under interventions
    def clear_reshaped_model_data(self):

        self.model.categorical_parents_reshaped = None
        self.model.continuous_parents_reshaped = None
        self.model.outcome_reshaped = None

        if hasattr(self, "predictive"):
            self.predictive.categorical_parents_reshaped = None
            self.predictive.continuous_parents_reshaped = None
            self.predictive.outcome_reshaped = None

    def train_model(self, num_steps=None, override=False):

        if not override:
            if os.path.isfile(self.guide_path) and os.path.isfile(self.param_path):
                print(
                    "The model has already been trained. Set 'override' to True to retrain the model."
                )
                return

        if num_steps is None:
            num_steps = self.num_steps

        pyro.clear_param_store()

        self.clear_reshaped_model_data()

        guide = run_svi_inference(
            self.model, n_steps=num_steps, lr=0.03, plot=False, data=self.data
        )

        serialized_guide = dill.dumps(guide)
        with open(self.guide_path, "wb") as file:
            file.write(serialized_guide)

        with open(self.param_path, "wb") as file:
            pyro.get_param_store().save(self.param_path)

    def predict_cumulative(self, conn, intervention):

        limit_intervention = self._tracts_intervention(conn, **intervention)
        intervention_year = intervention["reform_year"] - 2011

        self.clear_reshaped_model_data()

        with do(actions={"limit": limit_intervention}):
            intervened_samples = self.predictive(
                self.nonified_data, intervention_year=intervention_year
            )

        intervened_samples["destandardized_housing_units_cumulative"] = (
            intervened_samples["predicted_housing_units_cumulative"]
            * self.data["housing_units_cumulative_std"]
            + self.data["housing_units_cumulative_mean"]
        )

        intervened_summary = summarize_time_series(
            intervened_samples,
            self.observed_housing_cumulative,
            y_site="destandardized_housing_units_cumulative",
            clamp_at_zero=True,
            compute_metrics=False,
        )

        self.intervened_summary = intervened_summary

        return {
            "census_tracts": self.census_tracts,
            "years": self.years,
            "housing_units_observed": self.observed_housing_cumulative_list,
            "housing_units_factual_means": self.factual_means_list,
            "housing_units_factual_low": self.factual_low_list,
            "housing_units_factual_high": self.factual_high_list,
            "housing_units_factual_samples": self.factual_samples_list,
            "housing_units_intervened_means": self.convert_dict_to_list(
                intervened_summary["series_mean_pred"]
            ),
            "housing_units_intervened_low": self.convert_dict_to_list(
                intervened_summary["series_low_pred"]
            ),
            "housing_units_intervened_high": self.convert_dict_to_list(
                intervened_summary["series_high_pred"]
            ),
            "housing_units_intervened_samples": self.generate_samples_list(
                intervened_samples["destandardized_housing_units_cumulative"]
            ),
        }


# DON'T DELETE THIS INFORMATION:
# This the desired structure of the output on the front-end side
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
        instantiation_start = time.time()
        predictor = TractsModelPredictor(conn)
        instantiation_end = time.time()

        start = time.time()
        for iter in range(2):  # added for time testing
            result = predictor.predict_cumulative(
                conn,
                intervention={
                    "radius_blue": 106.7,
                    "limit_blue": 0,
                    "radius_yellow_line": 402.3,
                    "radius_yellow_stop": 804.7,
                    "limit_yellow": 0.5,
                    "reform_year": 2015,
                },
            )
        end = time.time()
        print(f"Instantiation in {instantiation_end - instantiation_start} seconds")
        print(f"2 predictions in {end - start} seconds")
