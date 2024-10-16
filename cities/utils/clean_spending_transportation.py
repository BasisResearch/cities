import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()


def clean_spending_transportation():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide
    gdp = gdp.get("gdp")

    spending_transportation = pd.read_csv(
        f"{root}/data/raw/spending_transportation.csv"
    )

    transportUnwanted = spending_transportation[
        (
            pd.isna(spending_transportation["total_obligated_amount"])
            | (spending_transportation["total_obligated_amount"] == 1)
            | (spending_transportation["total_obligated_amount"] == 0)
        )
    ]

    exclude_mask = spending_transportation["total_obligated_amount"].isin(
        transportUnwanted["total_obligated_amount"]
    )
    spending_transportation = spending_transportation[
        ~exclude_mask
    ]  # 66 values removed

    assert spending_transportation.isna().sum().sum() == 0, "Na values detected"

    # loading names and repearing fips of value 3 and shorter

    names_transportation = pd.read_csv(
        f"{root}/data/raw/spending_transportation_names.csv"
    )

    short_geofips = spending_transportation[
        spending_transportation["GeoFIPS"].astype(str).str.len().between(1, 3)
    ]

    spending_only_fips = np.setdiff1d(
        spending_transportation["GeoFIPS"], gdp["GeoFIPS"]
    )

    fips4_to_repeair = [
        fip for fip in spending_only_fips if (fip < 10000 and fip > 999)
    ]
    short4_fips = spending_transportation[
        spending_transportation["GeoFIPS"].isin(fips4_to_repeair)
    ]

    full_geofipsLIST = [fip for fip in spending_only_fips if fip > 9999]
    full_geofips = spending_transportation[
        spending_transportation["GeoFIPS"].isin(full_geofipsLIST)
    ]

    cleaningLIST = [full_geofips, short4_fips, short_geofips]

    for badFIPS in cleaningLIST:
        geofips_to_geonamealt = dict(
            zip(names_transportation["GeoFIPS"], names_transportation["GeoNameALT"])
        )

        badFIPS["GeoNameALT"] = badFIPS["GeoFIPS"].map(geofips_to_geonamealt)
        badFIPS = badFIPS.rename(columns={"GeoFIPS": "damagedFIPS"})

        badFIPSmapping_dict = dict(zip(gdp["GeoName"], gdp["GeoFIPS"]))

        badFIPS["repairedFIPS"] = badFIPS["GeoNameALT"].apply(
            lambda x: badFIPSmapping_dict.get(x)
        )
        repaired_geofips = badFIPS[badFIPS["repairedFIPS"].notna()]

        repair_ratio = repaired_geofips.shape[0] / badFIPS.shape[0]
        print(f"Ratio of repaired FIPS: {round(repair_ratio, 2)}")

        # assert repair_ratio > 0.9, f'Less than 0.9 of FIPS were successfully repaired!'

        spending_transportation["GeoFIPS"] = spending_transportation["GeoFIPS"].replace(
            dict(zip(repaired_geofips["damagedFIPS"], repaired_geofips["repairedFIPS"]))
        )

    # deleting short FIPS codes
    count_short_geofips = spending_transportation[
        spending_transportation["GeoFIPS"] <= 999
    ]["GeoFIPS"].count()
    assert (
        count_short_geofips / spending_transportation.shape[0] < 0.05
    ), "More than 0.05 of FIPS are short and will be deleted!"

    spending_transportation = spending_transportation[
        spending_transportation["GeoFIPS"] > 999
    ]

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), spending_transportation["GeoFIPS"].unique()
    )

    all_FIPS_spending_transportation = spending_transportation.copy()

    spending_transportation = spending_transportation[
        spending_transportation["GeoFIPS"].isin(common_fips)
    ]  # 0.96 of FIPS are common
    assert (
        spending_transportation.shape[0] / all_FIPS_spending_transportation.shape[0]
        > 0.9
    ), "Less than 0.9 of FIPS are common!"

    # grouping duplicate fips for years
    # (they appeared because we have repaired some of them and now they match with number that is already present)

    spending_transportation = (
        spending_transportation.groupby(["GeoFIPS", "year"])["total_obligated_amount"]
        .sum()
        .reset_index()
    )
    spending_transportation.reset_index(drop=True, inplace=True)

    # adding GeoNames
    spending_transportation = spending_transportation.merge(
        gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left"
    )[["GeoFIPS", "GeoName", "year", "total_obligated_amount"]]

    # adding missing FIPS with 0 values in total_obligated_amount column, and 2018 year (as a dummy variable)

    unique_gdp = gdp[["GeoFIPS", "GeoName"]].drop_duplicates(
        subset=["GeoFIPS", "GeoName"], keep="first"
    )
    exclude_geofips = set(spending_transportation["GeoFIPS"])
    unique_gdp = unique_gdp[~unique_gdp["GeoFIPS"].isin(exclude_geofips)]

    unique_gdp["year"] = np.repeat(2018, unique_gdp.shape[0])
    unique_gdp["total_obligated_amount"] = np.repeat(0, unique_gdp.shape[0])
    spending_transportation = pd.concat(
        [spending_transportation, unique_gdp], ignore_index=True
    )
    spending_transportation = spending_transportation.sort_values(
        by=["GeoFIPS", "GeoName", "year"]
    )

    assert (
        spending_transportation["GeoFIPS"].nunique()
        == spending_transportation["GeoName"].nunique()
    )
    assert spending_transportation["GeoFIPS"].nunique() == gdp["GeoFIPS"].nunique()

    spending_transportation = spending_transportation.rename(columns={"year": "Year"})

    # standardizing and saving
    spending_transportation_long = spending_transportation.copy()

    spending_transportation_wide = spending_transportation.pivot_table(
        index=["GeoFIPS", "GeoName"], columns="Year", values="total_obligated_amount"
    )
    spending_transportation_wide.reset_index(inplace=True)
    spending_transportation_wide.columns.name = None
    spending_transportation_wide = spending_transportation_wide.fillna(0)

    spending_transportation_std_long = standardize_and_scale(
        spending_transportation_long
    )
    spending_transportation_std_wide = standardize_and_scale(
        spending_transportation_wide
    )

    spending_transportation_wide.to_csv(
        f"{root}/data/processed/spending_transportation_wide.csv", index=False
    )
    spending_transportation_long.to_csv(
        f"{root}/data/processed/spending_transportation_long.csv", index=False
    )
    spending_transportation_std_wide.to_csv(
        f"{root}/data/processed/spending_transportation_std_wide.csv", index=False
    )
    spending_transportation_std_long.to_csv(
        f"{root}/data/processed/spending_transportation_std_long.csv", index=False
    )
