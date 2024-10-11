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

from cities.modeling.zoning_models.zoning_tracts_sqm_model import (
    TractsModelSqm as TractsModel,
)
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data, select_from_sql

load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))


class TractsModelPredictor:
    kwargs = {
        "categorical": ["year", "year_original", "census_tract"],
        "continuous": {
            "housing_units",
            "total_value",
            "median_value",
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
            "select * from tracts_model__census_tracts order by census_tract, year",
            conn,
            TractsModelPredictor.kwargs,
        )
        self.data["continuous"]["mean_limit_original"] = self.obs_limits(conn)
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

    def predict_cumulative_by_year(self, conn, intervention):
        """Predict the cumulative number of housing units built from 2011-2019 under intervention, by year.

        Returns a dictionary with keys:
        - 'census_tracts': the tracts considered
        - 'years': the years considered (2011-2019)
        - 'housing_units_factual': cumulative housing units built according to real housing data, by year
        - 'housing_units_counterfactual': samples from prediction of cumulative housing units built, by year
        """
        pyro.clear_param_store()
        pyro.get_param_store().load(self.param_path)

        subset_for_preds = copy.deepcopy(self.subset)
        subset_for_preds["continuous"]["housing_units"] = None

        limit_intervention = self._tracts_intervention(conn, **intervention)

        with MultiWorldCounterfactual() as mwc:
            with do(actions={"limit": limit_intervention}):
                result_all = self.predictive(**subset_for_preds)["housing_units"]
        with mwc:
            result = gather(
                result_all, IndexSet(**{"limit": {1}}), event_dims=0
            ).squeeze()

        years = self.data["categorical"]["year_original"]
        tracts = self.data["categorical"]["census_tract"]
        f_housing_units = self.data["continuous"]["housing_units_original"]
        cf_housing_units = result * self.housing_units_std + self.housing_units_mean

        # Organize cumulative data by year and tract
        f_data = {}
        cf_data = {}
        unique_years = sorted(set(years.tolist()))
        unique_years = [year for year in unique_years if year <= 2019]  # Exclude years after 2019
        unique_tracts = sorted(set(tracts.tolist()))

        for year in unique_years:
            f_data[year] = {tract: 0 for tract in unique_tracts}
            cf_data[year] = {tract: [0] * 100 for tract in unique_tracts}

        for i in range(tracts.shape[0]):
            year = years[i].item()
            if year > 2019:
                continue  # Skip data for years after 2019
            tract = tracts[i].item()
            
            # Update factual data
            for y in unique_years:
                if y >= year:
                    f_data[y][tract] += f_housing_units[i].item()

            # Update counterfactual data
            if year < intervention["reform_year"]:
                for y in unique_years:
                    if y >= year:
                        cf_data[y][tract] = [x + f_housing_units[i].item() for x in cf_data[y][tract]]
            else:
                for y in unique_years:
                    if y >= year:
                        cf_data[y][tract] = [x + y for x, y in zip(cf_data[y][tract], cf_housing_units[:, i].tolist())]

        # Convert to lists for easier JSON serialization
        housing_units_factual = [[f_data[year][tract] for tract in unique_tracts] for year in unique_years]
        housing_units_counterfactual = [[cf_data[year][tract] for tract in unique_tracts] for year in unique_years]

        return {
            "census_tracts": unique_tracts,
            "years": unique_years,
            "housing_units_factual": housing_units_factual,
            "housing_units_counterfactual": housing_units_counterfactual,
        }

if __name__ == "__main__":
    import time

    from cities.utils.data_loader import db_connection

    with db_connection() as conn:
        predictor = TractsModelPredictor(conn)
        start = time.time()

        for iter in range(5):  # added for time testing
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
        print(f"Counterfactual in {end - start} seconds")
