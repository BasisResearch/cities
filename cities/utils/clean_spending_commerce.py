import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber


def clean_spending_commerce():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide
    gdp = gdp.get("gdp")

    spending_commerce = pd.read_csv("../data/raw/spending_commerce.csv")

    transportUnwanted = spending_commerce[
        (
            pd.isna(spending_commerce["total_obligated_amount"])
            | (spending_commerce["total_obligated_amount"] == 1)
            | (spending_commerce["total_obligated_amount"] == 0)
        )
    ]

    exclude_mask = spending_commerce["total_obligated_amount"].isin(
        transportUnwanted["total_obligated_amount"]
    )
    spending_commerce = spending_commerce[~exclude_mask]  # 24 values lost

    assert spending_commerce.isna().sum().sum() == 0, "Na values detected"

    # loading names and repearing fips of value 3 and shorter

    names_commerce = pd.read_csv("../data/raw/spending_commerce_names.csv")

    spending_only_fips = np.setdiff1d(spending_commerce["GeoFIPS"], gdp["GeoFIPS"])

    fips4_to_repair = [fip for fip in spending_only_fips if (fip < 10000 and fip > 999)]
    short4_fips = spending_commerce[spending_commerce["GeoFIPS"].isin(fips4_to_repair)]

    full_geofipsLIST = [fip for fip in spending_only_fips if fip > 9999]
    full_geofips = spending_commerce[
        spending_commerce["GeoFIPS"].isin(full_geofipsLIST)
    ]

    cleaningLIST = [full_geofips, short4_fips]  # no small fips

    # replacing damaged FIPS

    for badFIPS in cleaningLIST:
        geofips_to_geonamealt = dict(
            zip(names_commerce["GeoFIPS"], names_commerce["GeoNameALT"])
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

        spending_commerce["GeoFIPS"] = spending_commerce["GeoFIPS"].replace(
            dict(zip(repaired_geofips["damagedFIPS"], repaired_geofips["repairedFIPS"]))
        )

    # deleting short FIPS codes

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), spending_commerce["GeoFIPS"].unique()
    )

    all_FIPS_spending_commerce = spending_commerce.copy()

    spending_commerce = spending_commerce[
        spending_commerce["GeoFIPS"].isin(common_fips)
    ]  # 67 FIPS deleted
    assert (
        spending_commerce.shape[0] / all_FIPS_spending_commerce.shape[0] > 0.9
    ), "Less than 0.9 of FIPS are common!"

    # grouping duplicate fips for years
    # (they appeared because we have repaired some of them and now they match with number that is already present)

    spending_commerce = (
        spending_commerce.groupby(["GeoFIPS", "year"])["total_obligated_amount"]
        .sum()
        .reset_index()
    )
    spending_commerce.reset_index(drop=True, inplace=True)

    # adding GeoNames
    spending_commerce = spending_commerce.merge(
        gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left"
    )[["GeoFIPS", "GeoName", "year", "total_obligated_amount"]]

    unique_gdp = gdp[["GeoFIPS", "GeoName"]].drop_duplicates(
        subset=["GeoFIPS", "GeoName"], keep="first"
    )
    exclude_geofips = set(spending_commerce["GeoFIPS"])
    unique_gdp = unique_gdp[~unique_gdp["GeoFIPS"].isin(exclude_geofips)]

    unique_gdp["year"] = np.repeat(2018, unique_gdp.shape[0])
    unique_gdp["total_obligated_amount"] = np.repeat(0, unique_gdp.shape[0])
    spending_commerce = pd.concat([spending_commerce, unique_gdp], ignore_index=True)
    spending_commerce = spending_commerce.sort_values(by=["GeoFIPS", "GeoName", "year"])

    assert (
        spending_commerce["GeoFIPS"].nunique() == spending_commerce["GeoName"].nunique()
    )
    assert spending_commerce["GeoFIPS"].nunique() == gdp["GeoFIPS"].nunique()

    spending_commerce = spending_commerce.rename(columns={'year': 'Year'})

    # standardizing and saving
    spending_commerce_long = spending_commerce.copy()

    spending_commerce_wide = spending_commerce.pivot_table(
        index=["GeoFIPS", "GeoName"], columns="Year", values="total_obligated_amount"
    )
    spending_commerce_wide.reset_index(inplace=True)
    spending_commerce_wide.columns.name = None
    spending_commerce_wide = spending_commerce_wide.fillna(0)

    spending_commerce_std_long = standardize_and_scale(spending_commerce)
    spending_commerce_std_wide = standardize_and_scale(spending_commerce_wide)

    spending_commerce_wide.to_csv(
        "../data/processed/spending_commerce_wide.csv", index=False
    )
    spending_commerce_long.to_csv(
        "../data/processed/spending_commerce_long.csv", index=False
    )
    spending_commerce_std_wide.to_csv(
        "../data/processed/spending_commerce_std_wide.csv", index=False
    )
    spending_commerce_std_long.to_csv(
        "../data/processed/spending_commerce_std_long.csv", index=False
    )
