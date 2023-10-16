import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber


def clean_transport():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide
    gdp = gdp.get("gdp")

    # grabbing gdp for comparison

    transport = pd.read_csv("../data/raw/smartLocationSmall.csv")

    # choosing transport variables
    transport = transport[["GeoFIPS", "D3A", "WeightAvgNatWalkInd"]]

    # list of GeoFips with Na values
    transportUnwanted = transport[
        (
            pd.isna(transport["WeightAvgNatWalkInd"])
            | (transport["WeightAvgNatWalkInd"] == 1)
        )
        | (transport["D3A"] == 0)
        | (transport["D3A"] == 1)
    ]

    exclude_mask = transport["GeoFIPS"].isin(transportUnwanted["GeoFIPS"])
    transport = transport[~exclude_mask]

    # the step above deleted 10 records with NAs,
    # no loss on a dataset because they were not common with gdp anyway

    assert transport.isna().sum().sum() == 0, "Na values detected"
    assert transport["GeoFIPS"].is_unique

    # subsetting to common FIPS numbers

    common_fips = np.intersect1d(gdp["GeoFIPS"].unique(), transport["GeoFIPS"].unique())
    transport = transport[transport["GeoFIPS"].isin(common_fips)]

    assert len(common_fips) == len(transport["GeoFIPS"].unique())
    assert len(transport) > 2800, "The number of records is lower than 3000"

    # adding geoname column
    transport = transport.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")[
        ["GeoFIPS", "GeoName", "D3A", "WeightAvgNatWalkInd"]
    ]

    # renaming D3A to roadDenisty
    transport.rename(columns={"D3A": "roadDensity"}, inplace=True)

    patState = r", [A-Z]{2}(\*{1,2})?$"
    GeoNameError = "Wrong GeoName value!"
    assert transport["GeoName"].str.contains(patState, regex=True).all(), GeoNameError
    assert sum(transport["GeoName"].str.count(", ")) == transport.shape[0], GeoNameError

    # changing values to floats

    for column in transport.columns[2:]:
        transport[column] = transport[column].astype(float)

    # Standardizing, formatting, saving

    transport_wide = transport.copy()
    transport_std_wide = standardize_and_scale(transport)

    transport_long = pd.melt(
        transport,
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )
    transport_std_long = pd.melt(
        transport_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    transport_wide.to_csv("../data/processed/transport_wide.csv", index=False)
    transport_long.to_csv("../data/processed/transport_long.csv", index=False)
    transport_std_wide.to_csv("../data/processed/transport_std_wide.csv", index=False)
    transport_std_long.to_csv("../data/processed/transport_std_long.csv", index=False)
