import numpy as np
import pandas as pd
from pathlib import Path
import os

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber

path = Path(__file__).parent.absolute()

def clean_unemployment():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    dtype_mapping = {"STATE": str, "COUNTY": str}
    unemployment_rate = pd.read_csv(os.path.join(path, "../../data/raw/unemployment_rate_wide_withNA.csv"), dtype=dtype_mapping)
    print('hello')
    
    unemployment_rate["GeoFIPS"] = unemployment_rate["GeoFIPS"].astype(int)

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), unemployment_rate["GeoFIPS"].unique()
    )

    unemployment_rate = unemployment_rate[unemployment_rate["GeoFIPS"].isin(common_fips)]
    
    unemployment_rate = unemployment_rate.merge(
        gdp[["GeoFIPS", "GeoName"]], on=["GeoFIPS", "GeoName"], how="outer"
    )
    print(unemployment_rate.head())
    unemployment_rate = unemployment_rate.sort_values(by=["GeoFIPS", "GeoName"])

    unemployment_rate_wide = unemployment_rate.copy()

    unemployment_rate_long = pd.melt(
        unemployment_rate,
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Year",
        value_name="Value",
    )

    unemployment_rate_std_wide = standardize_and_scale(unemployment_rate)

    unemployment_rate_std_long = pd.melt(
        unemployment_rate_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Year",
        value_name="Value",
    )

    unemployment_rate_wide.to_csv(os.path.join(path, "../../data/processed/unemployment_rate_wide.csv"), index=False)
    unemployment_rate_long.to_csv(os.path.join(path, "../../data/processed/unemployment_rate_long.csv"), index=False)
    unemployment_rate_std_wide.to_csv(os.path.join(path, "../../data/processed/unemployment_rate_std_wide.csv"), index=False)
    unemployment_rate_std_long.to_csv(os.path.join(path, "../../data/processed/unemployment_rate_std_long.csv"), index=False)
