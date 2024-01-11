import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

def clean_ethnic_composition():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    ethnic_composition = pd.read_csv(f"{root}/data/raw/ACSDP5Y2021_DP05_Race.csv")

    ethnic_composition = ethnic_composition.iloc[1:]
    ethnic_composition["GEO_ID"].isna() == 0

    ethnic_composition["GEO_ID"] = ethnic_composition["GEO_ID"].str.split("US").str[1]
    ethnic_composition["GEO_ID"] = ethnic_composition["GEO_ID"].astype("int64")
    ethnic_composition = ethnic_composition.rename(columns={"GEO_ID": "GeoFIPS"})

    ethnic_composition = ethnic_composition[
        ["GeoFIPS"] + [col for col in ethnic_composition.columns if col.endswith("E")]
    ]
    ethnic_composition = ethnic_composition.drop(columns=["NAME"])

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), ethnic_composition["GeoFIPS"].unique()
    )
    len(common_fips)

    ethnic_composition = ethnic_composition[
        ethnic_composition["GeoFIPS"].isin(common_fips)
    ]

    ethnic_composition = ethnic_composition.merge(
        gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left"
    )

    ethnic_composition = ethnic_composition[
        [
            "GeoFIPS",
            "GeoName",
            "DP05_0070E",
            "DP05_0072E",
            "DP05_0073E",
            "DP05_0074E",
            "DP05_0075E",
            "DP05_0077E",
            "DP05_0078E",
            "DP05_0079E",
            "DP05_0080E",
            "DP05_0081E",
            "DP05_0082E",
            "DP05_0083E",
        ]
    ]

    ethnic_composition.columns = [
        "GeoFIPS",
        "GeoName",
        "total_pop",
        "mexican",
        "puerto_rican",
        "cuban",
        "other_hispanic_latino",
        "white",
        "black_african_american",
        "american_indian_alaska_native",
        "asian",
        "native_hawaiian_other_pacific_islander",
        "other_race",
        "two_or_more_sum",
    ]
    ethnic_composition = ethnic_composition.sort_values(by=["GeoFIPS", "GeoName"])

    ethnic_composition.iloc[:, 2:] = ethnic_composition.iloc[:, 2:].apply(
        pd.to_numeric, errors="coerce"
    )
    ethnic_composition[ethnic_composition.columns[2:]] = ethnic_composition[
        ethnic_composition.columns[2:]
    ].astype(float)

    ethnic_composition["other_race_races"] = (
        ethnic_composition["other_race"] + ethnic_composition["two_or_more_sum"]
    )
    ethnic_composition = ethnic_composition.drop(
        ["other_race", "two_or_more_sum"], axis=1
    )

    ethnic_composition["totalALT"] = ethnic_composition.iloc[:, 3:].sum(axis=1)
    assert (ethnic_composition["totalALT"] == ethnic_composition["total_pop"]).all()
    ethnic_composition = ethnic_composition.drop("totalALT", axis=1)

    # copy with nominal values
    ethnic_composition.to_csv(f"{root}/data/raw/ethnic_composition_nominal.csv", index=False)

    row_sums = ethnic_composition.iloc[:, 2:].sum(axis=1)
    ethnic_composition.iloc[:, 3:] = ethnic_composition.iloc[:, 3:].div(
        row_sums, axis=0
    )

    ethnic_composition = ethnic_composition.drop(["total_pop"], axis=1)

    ethnic_composition_wide = ethnic_composition.copy()

    ethnic_composition_long = pd.melt(
        ethnic_composition,
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    ethnic_composition_std_wide = standardize_and_scale(ethnic_composition)

    ethnic_composition_std_long = pd.melt(
        ethnic_composition_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    ethnic_composition_wide.to_csv(
        f"{root}/data/processed/ethnic_composition_wide.csv", index=False
    )
    ethnic_composition_long.to_csv(
        f"{root}/data/processed/ethnic_composition_long.csv", index=False
    )
    ethnic_composition_std_wide.to_csv(
        f"{root}/data/processed/ethnic_composition_std_wide.csv", index=False
    )
    ethnic_composition_std_long.to_csv(
        f"{root}/data/processed/ethnic_composition_std_long.csv", index=False
    )
