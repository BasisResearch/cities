import numpy as np
import pandas as pd

from cities.utils.clean_variable import clean_variable, communities_tracts_to_counties
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

data = DataGrabber()
data.get_features_wide(["gdp"])
gdp = data.wide["gdp"]


def clean_burdens_first():
    burdens = pd.read_csv(f"{root}/data/raw/communities_raw.csv")

    list_variables = ["Housing burden (percent)", "Energy burden"]
    burdens = communities_tracts_to_counties(burdens, list_variables)

    burdens["GeoFIPS"] = burdens["GeoFIPS"].astype(np.int64)

    common_fips = np.intersect1d(burdens["GeoFIPS"].unique(), gdp["GeoFIPS"].unique())
    burdens = burdens[burdens["GeoFIPS"].isin(common_fips)]
    burdens = burdens.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")

    burdens = burdens[
        ["GeoFIPS", "GeoName", "Housing burden (percent)", "Energy burden"]
    ]

    burdens.columns = ["GeoFIPS", "GeoName", "Housing_burden", "Energy_burden"]

    columns_to_trans = burdens.columns[-2:]
    burdens[columns_to_trans] = burdens[columns_to_trans].astype("float64")

    burdens.to_csv(f"{root}/data/raw/burdens_raw.csv", index=False)


def clean_burdens():
    clean_burdens_first()

    variable_name = "burdens"
    path_to_raw_csv = f"{root}/data/raw/burdens_raw.csv"

    clean_variable(variable_name, path_to_raw_csv)
