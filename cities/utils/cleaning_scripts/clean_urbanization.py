import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()


def clean_urbanization():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    dtype_mapping = {"STATE": str, "COUNTY": str}
    urbanization = pd.read_csv(
        f"{root}/data/raw/2020_UA_COUNTY.csv", dtype=dtype_mapping
    )

    urbanization["GeoFIPS"] = urbanization["STATE"].astype(str) + urbanization[
        "COUNTY"
    ].astype(str)
    urbanization["GeoFIPS"] = urbanization["GeoFIPS"].astype(int)

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), urbanization["GeoFIPS"].unique()
    )

    urbanization = urbanization[urbanization["GeoFIPS"].isin(common_fips)]

    urbanization = urbanization.merge(
        gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left"
    )

    urbanization = urbanization[
        [
            "GeoFIPS",
            "GeoName",
            "POPDEN_RUR",
            "POPDEN_URB",
            "HOUDEN_COU",
            "HOUDEN_RUR",
            "ALAND_PCT_RUR",
        ]
    ]

    urbanization = urbanization.sort_values(by=["GeoFIPS", "GeoName"])

    urbanization_wide = urbanization.copy()

    urbanization_long = pd.melt(
        urbanization,
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    urbanization_std_wide = standardize_and_scale(urbanization)

    urbanization_std_long = pd.melt(
        urbanization_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    urbanization_wide.to_csv(
        f"{root}/data/processed/urbanization_wide.csv", index=False
    )
    urbanization_long.to_csv(
        f"{root}/data/processed/urbanization_long.csv", index=False
    )
    urbanization_std_wide.to_csv(
        f"{root}/data/processed/urbanization_std_wide.csv", index=False
    )
    urbanization_std_long.to_csv(
        f"{root}/data/processed/urbanization_std_long.csv", index=False
    )
