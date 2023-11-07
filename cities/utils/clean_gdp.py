import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale


def clean_gdp():
    gdp = pd.read_csv("../data/raw/CAGDP1_2001_2021.csv", encoding="ISO-8859-1")

    gdp = gdp.loc[:9533]  # drop notes at the bottom

    gdp["GeoFIPS"] = gdp["GeoFIPS"].fillna("").astype(str)
    gdp["GeoFIPS"] = gdp["GeoFIPS"].str.strip(' "').astype(int)

    # remove large regions
    gdp = gdp[gdp["GeoFIPS"] % 1000 != 0]

    # focus on chain-type GDP
    mask = gdp["Description"].str.startswith("Chain")
    gdp = gdp[mask]

    # drop Region number, Tablename, LineCode, IndustryClassification columns (the last one is empty anyway)
    gdp = gdp.drop(gdp.columns[2:8], axis=1)

    # 2012 makes no sense, it's 100 throughout
    gdp = gdp.drop("2012", axis=1)

    gdp.replace("(NA)", np.nan, inplace=True)
    gdp.replace("(NM)", np.nan, inplace=True)

    # nan_rows = gdp[gdp.isna().any(axis=1)] #  if inspection is needed

    gdp.dropna(axis=0, inplace=True)

    for column in gdp.columns[2:]:
        gdp[column] = gdp[column].astype(float)

    assert gdp["GeoName"].is_unique

    # subsetting GeoFIPS to values in exclusions.csv

    exclusions_df = pd.read_csv("../data/raw/exclusions.csv")
    gdp = gdp[~gdp["GeoFIPS"].isin(exclusions_df["exclusions"])]

    assert len(gdp) == len(gdp["GeoFIPS"].unique())
    assert len(gdp) > 2800, "The number of records is lower than 2800"

    patState = r", [A-Z]{2}(\*{1,2})?$"
    GeoNameError = "Wrong Geoname value!"
    assert gdp["GeoName"].str.contains(patState, regex=True).all(), GeoNameError
    assert sum(gdp["GeoName"].str.count(", ")) == gdp.shape[0], GeoNameError

    for column in gdp.columns[2:]:
        assert (gdp[column] > 0).all(), f"Negative values in {column}"
        assert gdp[column].isna().sum() == 0, f"Missing values in {column}"
        assert gdp[column].isnull().sum() == 0, f"Null values in {column}"
        assert (gdp[column] < 3000).all(), f"Values suspiciously large in {column}"

    # TODO_Nikodem investigate strange large values

    gdp_wide = gdp.copy()
    gdp_long = pd.melt(
        gdp.copy(), id_vars=["GeoFIPS", "GeoName"], var_name="Year", value_name="Value"
    )

    gdp_std_wide = standardize_and_scale(gdp)
    gdp_std_long = pd.melt(
        gdp_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Year",
        value_name="Value",
    )

    gdp_wide.to_csv("../data/processed/gdp_wide.csv", index=False)
    gdp_long.to_csv("../data/processed/gdp_long.csv", index=False)
    gdp_std_wide.to_csv("../data/processed/gdp_std_wide.csv", index=False)
    gdp_std_long.to_csv("../data/processed/gdp_std_long.csv", index=False)
