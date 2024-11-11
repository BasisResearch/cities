import os

import pandas as pd
import torch
import pyro
from dotenv import load_dotenv
import dill


from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather
from chirho.interventional.handlers import do

from cities.modeling.svi_inference import run_svi_inference
from cities.modeling.zoning_models.zoning_tracts_ts_model  import TractsModelCumulativeAR1 as TractsModel
from cities.utils.data_grabber import find_repo_root
from cities.modeling.zoning_models.ts_model_components import prepare_zoning_data_for_ts
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
num_steps = 300  # 1500

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
        housing_units_continuous_parents_names= ["median_value", "distance",
         "income", "white", "limit", "segregation", "sqm", "downtown_overlap", "university_overlap"],
        housing_units_continuous_interaction_pairs = self.ins,
        leeway= 0.9
        )

        self.root = find_repo_root()

        self.deploy_path = os.path.join(self.root, "cities/deployment/tracts_minneapolis")
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
            print("The model has no access to training results. Please run the 'train_model()' to generate the required files.)")

        else:
            pyro.clear_param_store()

            with open(self.guide_path, "rb") as file:
                self.guide = dill.load(file)

            pyro.get_param_store().load(self.param_path)

            self.predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=self.num_samples)
            


     

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

        self.predictive.categorical_parents_reshaped = None
        self.predictive.continuous_parents_reshaped = None
        self.predictive.outcome_reshaped = None
    
    def train_model(self, num_steps = None, override = False):

        if not override:
            if os.path.isfile(self.guide_path) and os.path.isfile(self.param_path):
                print("The model has already been trained. Set 'override' to True to retrain the model.")
                return

                
        if num_steps is None:
            num_steps = self.num_steps

        pyro.clear_param_store()

        self.clear_reshaped_model_data()

        guide = run_svi_inference(self.model, n_steps=num_steps, lr=0.03, plot=False, data = self.data)

        serialized_guide = dill.dumps(guide)
        with open(self.guide_path, "wb") as file:
            file.write(serialized_guide)

        with open(self.param_path, "wb") as file:
            pyro.get_param_store().save(self.param_path)



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
                result_all = self.predictive(data = self.nonified_data)#["housing_units"]

        return {
            "all_samples": result_all, "mwc": mwc
        }



