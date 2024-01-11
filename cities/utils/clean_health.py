import numpy as np
import pandas as pd

from cities.utils.clean_variable import clean_variable, communities_tracts_to_counties
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

data = DataGrabber()
data.get_features_wide(["gdp"])
gdp = data.wide["gdp"]


def clean_health_first():
    health = pd.read_csv(f"{root}/data/raw/communities_raw.csv")

    list_variables = [
        "Life expectancy (years)",
        "Current asthma among adults aged greater than or equal to 18 years",
        "Diagnosed diabetes among adults aged greater than or equal to 18 years",
        "Coronary heart disease among adults aged greater than or equal to 18 years",
    ]

    health = communities_tracts_to_counties(health, list_variables)

    health.dropna(inplace=True)

    health["GeoFIPS"] = health["GeoFIPS"].astype(np.int64)

    common_fips = np.intersect1d(health["GeoFIPS"].unique(), gdp["GeoFIPS"].unique())
    health = health[health["GeoFIPS"].isin(common_fips)]
    health = health.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")

    health = health[
        [
            "GeoFIPS",
            "GeoName",
            "Life expectancy (years)",
            "Current asthma among adults aged greater than or equal to 18 years",
            "Diagnosed diabetes among adults aged greater than or equal to 18 years",
            "Coronary heart disease among adults aged greater than or equal to 18 years",
        ]
    ]

    health.columns = [
        "GeoFIPS",
        "GeoName",
        "LifeExpectancy",
        "Asthma",
        "Diabetes",
        "HeartDisease",
    ]

    columns_to_round = health.columns[-3:]
    health[columns_to_round] = health[columns_to_round].round(0).astype("float64")
    health["LifeExpectancy"] = health["LifeExpectancy"].round(2).astype("float64")

    val_list = ["Asthma", "Diabetes", "HeartDisease"]

    for val in val_list:  # dealing with weird format of percentages
        health[val] = health[val] / 100

    health.to_csv(f"{root}/data/raw/health_raw.csv", index=False)


def clean_health():
    clean_health_first()

    variable_name = "health"
    path_to_raw_csv = f"{root}/data/raw/health_raw.csv"

    clean_variable(variable_name, path_to_raw_csv)
