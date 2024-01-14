import numpy as np
import pandas as pd

from cities.utils.clean_variable import VariableCleaner, communities_tracts_to_counties
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

    burdens.columns = ["GeoFIPS", "GeoName", "burdens_housing", "burdens_energy"]

    columns_to_trans = burdens.columns[-2:]
    burdens[columns_to_trans] = burdens[columns_to_trans].astype("float64")
    
    burdens_housing = burdens[["GeoFIPS", "GeoName", "burdens_housing"]]
    burdens_energy = burdens[["GeoFIPS", "GeoName", "burdens_energy"]]
    
    burdens_housing.to_csv(f"{root}/data/raw/burdens_housing_raw.csv", index=False)
    burdens_energy.to_csv(f"{root}/data/raw/burdens_energy_raw.csv", index=False)



def clean_burdens():
    clean_burdens_first()
    

    cleaner_housing = VariableCleaner(
        variable_name="burdens_housing",
        path_to_raw_csv=f"{root}/data/raw/burdens_housing_raw.csv",
        year_or_category="Category",
    )
    cleaner_housing.clean_variable()
    
    
    cleaner_energy = VariableCleaner(
        variable_name="burdens_energy",
        path_to_raw_csv=f"{root}/data/raw/burdens_energy_raw.csv",
        year_or_category="Category",
    )
    cleaner_energy.clean_variable()