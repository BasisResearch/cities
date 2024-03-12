import numpy as np
import pandas as pd

from cities.utils.clean_variable import (VariableCleaner,
                                         communities_tracts_to_counties)
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

data = DataGrabber()
data.get_features_wide(["gdp"])
gdp = data.wide["gdp"]


variables_hazard = [
    "expected_agricultural_loss_rate",
    "expected_building_loss_rate",
    "expected_population_loss_rate",
    "diesel_matter_exposure",
    "proximity_to_hazardous_waste_sites",
    "proximity_to_risk_management_plan_facilities",
]


def clean_hazard_first():
    hazard = pd.read_csv(f"{root}/data/raw/communities_raw.csv")

    list_variables = [
        "Expected agricultural loss rate (Natural Hazards Risk Index)",
        "Expected building loss rate (Natural Hazards Risk Index)",
        "Expected population loss rate (Natural Hazards Risk Index)",
        "Diesel particulate matter exposure",
        "Proximity to hazardous waste sites",
        "Proximity to Risk Management Plan (RMP) facilities",
    ]

    hazard = communities_tracts_to_counties(hazard, list_variables)

    hazard.dropna(inplace=True)

    hazard["GeoFIPS"] = hazard["GeoFIPS"].astype(np.int64)

    common_fips = np.intersect1d(hazard["GeoFIPS"].unique(), gdp["GeoFIPS"].unique())
    hazard = hazard[hazard["GeoFIPS"].isin(common_fips)]
    hazard = hazard.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")

    hazard = hazard[
        [
            "GeoFIPS",
            "GeoName",
            "Expected agricultural loss rate (Natural Hazards Risk Index)",
            "Expected building loss rate (Natural Hazards Risk Index)",
            "Expected population loss rate (Natural Hazards Risk Index)",
            "Diesel particulate matter exposure",
            "Proximity to hazardous waste sites",
            "Proximity to Risk Management Plan (RMP) facilities",
        ]
    ]

    hazard.columns = [
        "GeoFIPS",
        "GeoName",
        "expected_agricultural_loss_rate",
        "expected_building_loss_rate",
        "expected_population_loss_rate",
        "diesel_matter_exposure",
        "proximity_to_hazardous_waste_sites",
        "proximity_to_risk_management_plan_facilities",
    ]

    columns_to_trans = hazard.columns[-6:]
    hazard[columns_to_trans] = hazard[columns_to_trans].astype("float64")

    for variable in variables_hazard:
        hazard_variable = hazard[["GeoFIPS", "GeoName", variable]]
        hazard_variable.to_csv(f"{root}/data/raw/{variable}.csv", index=False)


def clean_hazard():
    clean_hazard_first()

    for variable in variables_hazard:
        cleaner = VariableCleaner(
            variable_name=variable,
            path_to_raw_csv=f"{root}/data/raw/{variable}.csv",
            year_or_category="Category",
        )
        cleaner.clean_variable()
