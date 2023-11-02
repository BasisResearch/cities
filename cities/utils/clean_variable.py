import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cities.utils.clean_gdp import clean_gdp
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber

path = Path(__file__).parent.absolute()


def clean_variable(variable_name, path_to_raw_csv, YearOrCategory="Year"):
    # function for cleaning a generic timeseries csv, wide format with these columns:
    # GeoFIPS, GeoName, 2001, 2002, 2003, 2004, 2005, 2006, 2007, ...

    # load gdb, to get list of current non-excluded FIPS codes
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    # load raw csv
    variable_db = pd.read_csv(path_to_raw_csv)
    variable_db["GeoFIPS"] = variable_db["GeoFIPS"].astype(int)

    # drop nans
    variable_db = variable_db.dropna()

    # check if there are any counties that are missing from unempl but in gdp
    # if so, add them to exclusions, and re-run gdp with new exclusions
    if len(np.setdiff1d(gdp["GeoFIPS"].unique(), variable_db["GeoFIPS"].unique())) > 0:
        # add new exclusions
        new_exclusions = np.setdiff1d(
            gdp["GeoFIPS"].unique(), variable_db["GeoFIPS"].unique()
        )
        print("Adding new exclusions to exclusions.pkl: " + str(new_exclusions))
        # open exclusions file
        with open("../data/raw/exclusions.pkl", "rb") as file:
            exclusions = pickle.load(file)
        exclusions["transport"] = np.append(exclusions["transport"], new_exclusions)
        exclusions["transport"] = np.unique(exclusions["transport"])
        with open("../data/raw/exclusions.pkl", "wb") as file:
            pickle.dump(exclusions, file)
        print("Rerunning gdp cleaning with new exclusions")
        # rerun gdp cleaning
        clean_gdp()
        clean_variable(variable_name, path_to_raw_csv)
        return

    # restrict to only common FIPS codes
    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), variable_db["GeoFIPS"].unique()
    )
    variable_db = variable_db[variable_db["GeoFIPS"].isin(common_fips)]
    variable_db = variable_db.merge(
        gdp[["GeoFIPS", "GeoName"]], on=["GeoFIPS", "GeoName"], how="left"
    )
    variable_db = variable_db.sort_values(by=["GeoFIPS", "GeoName"])

    # make sure that it passes this test data.wide[feature][column].dtype == float
    for column in variable_db.columns:
        if column not in ["GeoFIPS", "GeoName"]:
            variable_db[column] = variable_db[column].astype(float)

    # save 4 formats to .csv
    variable_db_wide = variable_db.copy()
    variable_db_long = pd.melt(
        variable_db,
        id_vars=["GeoFIPS", "GeoName"],
        var_name=YearOrCategory,
        value_name="Value",
    )
    variable_db_std_wide = standardize_and_scale(variable_db)
    variable_db_std_long = pd.melt(
        variable_db_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name=YearOrCategory,
        value_name="Value",
    )
    variable_db_wide.to_csv(
        os.path.join(path, "../../data/processed/" + variable_name + "_wide.csv"),
        index=False,
    )
    variable_db_long.to_csv(
        os.path.join(path, "../../data/processed/" + variable_name + "_long.csv"),
        index=False,
    )
    variable_db_std_wide.to_csv(
        os.path.join(path, "../../data/processed/" + variable_name + "_std_wide.csv"),
        index=False,
    )
    variable_db_std_long.to_csv(
        os.path.join(path, "../../data/processed/" + variable_name + "_std_long.csv"),
        index=False,
    )
