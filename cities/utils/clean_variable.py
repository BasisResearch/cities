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

    exclusions_df = pd.read_csv("../../data/raw/exclusions.csv")

    # Check if there are any counties that are missing from variable_db but in exclusions_df
    # If so, add them to exclusions, and re-run variable_db with new exclusions
    if len(np.setdiff1d(exclusions_df["exclusions"].unique(), variable_db["GeoFIPS"].unique())) > 0:
        # Add new exclusions
        new_exclusions = np.setdiff1d(
            exclusions_df["exclusions"].unique(), variable_db["GeoFIPS"].unique()
        )
        print("Adding new exclusions to exclusions.csv: " + str(new_exclusions))
        
        # Create a new DataFrame with the additional exclusions
        new_exclusions_df = pd.DataFrame({"dataset": [variable_name] * len(new_exclusions), "exclusions": new_exclusions})
        
        # Concatenate the new exclusions DataFrame with the existing exclusions DataFrame
        exclusions_df = pd.concat([exclusions_df, new_exclusions_df], ignore_index=True)
        
        # Save the updated exclusions back to the CSV file
        exclusions_df.to_csv("../../data/raw/exclusions.csv", index=False)

        print("Rerunning variable_db cleaning with new exclusions")

        clean_gdp()
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
