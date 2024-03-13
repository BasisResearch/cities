import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()


def clean_spending_HHS():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide
    gdp = gdp.get("gdp")

    spending_HHS = pd.read_csv(f"{root}/data/raw/spending_HHS.csv")

    transportUnwanted = spending_HHS[
        (
            pd.isna(spending_HHS["total_obligated_amount"])
            | (spending_HHS["total_obligated_amount"] == 1)
            | (spending_HHS["total_obligated_amount"] == 0)
        )
    ]

    exclude_mask = spending_HHS["total_obligated_amount"].isin(
        transportUnwanted["total_obligated_amount"]
    )
    spending_HHS = spending_HHS[~exclude_mask]  # 95 observations dleted

    assert spending_HHS.isna().sum().sum() == 0, "Na values detected"

    # loading names and repearing fips of value 3 and shorter

    names_HHS = pd.read_csv(f"{root}/data/raw/spending_HHS_names.csv")

    spending_only_fips = np.setdiff1d(spending_HHS["GeoFIPS"], gdp["GeoFIPS"])

    fips4_to_repair = [fip for fip in spending_only_fips if (fip < 10000 and fip > 999)]
    short4_fips = spending_HHS[spending_HHS["GeoFIPS"].isin(fips4_to_repair)]

    full_geofipsLIST = [fip for fip in spending_only_fips if fip > 9999]
    full_geofips = spending_HHS[spending_HHS["GeoFIPS"].isin(full_geofipsLIST)]

    cleaningLIST = [full_geofips, short4_fips]  # no 3digit FIPS

    # replacing damaged FIPS

    for badFIPS in cleaningLIST:
        geofips_to_geonamealt = dict(zip(names_HHS["GeoFIPS"], names_HHS["GeoNameALT"]))

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

        spending_HHS["GeoFIPS"] = spending_HHS[
            "GeoFIPS"
        ].replace(  # no FIPS were repaired actually
            dict(zip(repaired_geofips["damagedFIPS"], repaired_geofips["repairedFIPS"]))
        )

    common_fips = np.intersect1d(
        gdp["GeoFIPS"].unique(), spending_HHS["GeoFIPS"].unique()
    )

    all_FIPS_spending_HHS = spending_HHS.copy()

    spending_HHS = spending_HHS[
        spending_HHS["GeoFIPS"].isin(common_fips)
    ]  # 99 FIPS deleted
    assert (
        spending_HHS.shape[0] / all_FIPS_spending_HHS.shape[0] > 0.9
    ), "Less than 0.9 of FIPS are common!"

    # grouping duplicate fips for years
    # (they appeared because we have repaired some of them and now they match with number that is already present)

    spending_HHS = (
        spending_HHS.groupby(["GeoFIPS", "year"])["total_obligated_amount"]
        .sum()
        .reset_index()
    )
    spending_HHS.reset_index(drop=True, inplace=True)

    # adding GeoNames
    spending_HHS = spending_HHS.merge(
        gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left"
    )[["GeoFIPS", "GeoName", "year", "total_obligated_amount"]]

    unique_gdp = gdp[["GeoFIPS", "GeoName"]].drop_duplicates(
        subset=["GeoFIPS", "GeoName"], keep="first"
    )
    exclude_geofips = set(spending_HHS["GeoFIPS"])
    unique_gdp = unique_gdp[~unique_gdp["GeoFIPS"].isin(exclude_geofips)]

    unique_gdp["year"] = np.repeat(2018, unique_gdp.shape[0])
    unique_gdp["total_obligated_amount"] = np.repeat(0, unique_gdp.shape[0])
    spending_HHS = pd.concat([spending_HHS, unique_gdp], ignore_index=True)
    spending_HHS = spending_HHS.sort_values(by=["GeoFIPS", "GeoName", "year"])

    assert spending_HHS["GeoFIPS"].nunique() == spending_HHS["GeoName"].nunique()
    assert spending_HHS["GeoFIPS"].nunique() == gdp["GeoFIPS"].nunique()

    # Assuming you have a DataFrame named 'your_dataframe'
    spending_HHS = spending_HHS.rename(columns={"year": "Year"})

    # standardizing and saving
    spending_HHS_long = spending_HHS.copy()

    spending_HHS_wide = spending_HHS.pivot_table(
        index=["GeoFIPS", "GeoName"], columns="Year", values="total_obligated_amount"
    )
    spending_HHS_wide.reset_index(inplace=True)
    spending_HHS_wide.columns.name = None
    spending_HHS_wide = spending_HHS_wide.fillna(0)

    spending_HHS_std_long = standardize_and_scale(spending_HHS)
    spending_HHS_std_wide = standardize_and_scale(spending_HHS_wide)

    spending_HHS_wide.to_csv(
        f"{root}/data/processed/spending_HHS_wide.csv", index=False
    )
    spending_HHS_long.to_csv(
        f"{root}/data/processed/spending_HHS_long.csv", index=False
    )
    spending_HHS_std_wide.to_csv(
        f"{root}/data/processed/spending_HHS_std_wide.csv", index=False
    )
    spending_HHS_std_long.to_csv(
        f"{root}/data/processed/spending_HHS_std_long.csv", index=False
    )
