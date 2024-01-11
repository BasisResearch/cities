from pathlib import Path

import numpy as np
import pandas as pd

from cities.utils.clean_variable import VariableCleaner
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber, find_repo_root

root = find_repo_root()

path = Path(__file__).parent.absolute()


def clean_industry_step_one():
    data = DataGrabber()
    data.get_features_wide(["gdp"])
    gdp = data.wide["gdp"]

    industry = pd.read_csv(f"{root}/data/raw/ACSDP5Y2021_DP03_industry.csv")

    industry["GEO_ID"] = industry["GEO_ID"].str.split("US").str[1]
    industry["GEO_ID"] = industry["GEO_ID"].astype("int64")
    industry = industry.rename(columns={"GEO_ID": "GeoFIPS"})

    common_fips = np.intersect1d(gdp["GeoFIPS"].unique(), industry["GeoFIPS"].unique())

    industry = industry[industry["GeoFIPS"].isin(common_fips)]

    industry = industry.merge(gdp[["GeoFIPS", "GeoName"]], on="GeoFIPS", how="left")

    industry = industry[
        [
            "GeoFIPS",
            "GeoName",
            "DP03_0004E",
            "DP03_0033E",
            "DP03_0034E",
            "DP03_0035E",
            "DP03_0036E",
            "DP03_0037E",
            "DP03_0038E",
            "DP03_0039E",
            "DP03_0040E",
            "DP03_0041E",
            "DP03_0042E",
            "DP03_0043E",
            "DP03_0044E",
            "DP03_0045E",
        ]
    ]

    column_name_mapping = {
        "DP03_0004E": "employed_sum",
        "DP03_0033E": "agri_forestry_mining",
        "DP03_0034E": "construction",
        "DP03_0035E": "manufacturing",
        "DP03_0036E": "wholesale_trade",
        "DP03_0037E": "retail_trade",
        "DP03_0038E": "transport_utilities",
        "DP03_0039E": "information",
        "DP03_0040E": "finance_real_estate",
        "DP03_0041E": "prof_sci_mgmt_admin",
        "DP03_0042E": "education_health",
        "DP03_0043E": "arts_entertainment",
        "DP03_0044E": "other_services",
        "DP03_0045E": "public_admin",
    }

    industry.rename(columns=column_name_mapping, inplace=True)

    industry = industry.sort_values(by=["GeoFIPS", "GeoName"])

    industry.to_csv(f"{root}/data/raw/industry_absolute.csv", index=False)

    row_sums = industry.iloc[:, 3:].sum(axis=1)

    industry.iloc[:, 3:] = industry.iloc[:, 3:].div(row_sums, axis=0)
    industry = industry.drop(["employed_sum"], axis=1)

    industry.to_csv(f"{root}/data/raw/industry_percent.csv", index=False)

    industry_wide = industry.copy()

    industry_long = pd.melt(
        industry,
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    industry_std_wide = standardize_and_scale(industry)

    industry_std_long = pd.melt(
        industry_std_wide.copy(),
        id_vars=["GeoFIPS", "GeoName"],
        var_name="Category",
        value_name="Value",
    )

    industry_wide.to_csv(f"{root}/data/processed/industry_wide.csv", index=False)
    industry_long.to_csv(f"{root}/data/processed/industry_long.csv", index=False)
    industry_std_wide.to_csv(f"{root}/data/processed/industry_std_wide.csv", index=False)
    industry_std_long.to_csv(f"{root}/data/processed/industry_std_long.csv", index=False)


def clean_industry():
    clean_industry_step_one()

    cleaner = VariableCleaner(
        variable_name="industry",
        path_to_raw_csv=f"{root}/data/raw/industry_percent.csv",
    )
    cleaner.clean_variable()
