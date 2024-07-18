import numpy as np
import pandas as pd
import requests
from us import states

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()
variables = (
    "NAME,"
    "DP03_0004E,"
    "DP03_0033E,"
    "DP03_0034E,"
    "DP03_0035E,"
    "DP03_0036E,"
    "DP03_0037E,"
    "DP03_0038E,"
    "DP03_0039E,"
    "DP03_0040E,"
    "DP03_0041E,"
    "DP03_0042E,"
    "DP03_0043E,"
    "DP03_0044E,"
    "DP03_0045E"
)


county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html

interval = [2019, 2022]
dfs = []

for year in interval:
    for x in range(
        0, len(states.STATES)
    ):  # in this call it's not possible to use the '*' wildcard to access all states, so we need to iterate over all states
        fips = states.STATES[x].fips

        url = (
            f"https://api.census.gov/data/{year}/acs/acs5/profile?"
            f"get={variables}&for=tract:{tract}&in=state:{fips}&in=county:{county_fips}&key={api_key}"
        )

        response = requests.get(url)

        assert (
            response.status_code == 200
        ), "The data retrieval went wrong"  # 200 means success

        print(f"{fips} fips for year {year} done")

        data = response.json()

        df = pd.DataFrame(data[1:], columns=data[0])
        df["Year"] = year  # Add the year column

        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


industry = combined_df.copy()

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

state_abbreviations = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

industry["GeoFIPS"] = industry.apply(
    lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1
).astype(np.int64)

industry.drop(["state", "county", "tract"], axis=1, inplace=True)


def parse_geo_name(name):
    if ";" in name:
        parts = name.split(";")
    else:
        parts = name.split(",")

    if len(parts) >= 3:
        county = parts[1].strip().replace(" County", "")
        state_full = parts[2].strip()
        state_abbr = state_abbreviations.get(state_full, state_full)
        return f"{county}, {state_abbr} (CT)"
    return "Unknown"


industry["GeoName"] = industry["NAME"].apply(parse_geo_name).astype(str)

assert (
    industry[industry["GeoName"] == "Unknown"].shape[0] == 0
), "There are Unknown GeoNames"

industry = industry.drop(["NAME"], axis=1)


rows1 = industry.shape[0]
industry.dropna(how="any", inplace=True)  # Drop NaN values inplace
rows2 = industry.shape[0]
print(f"This many rows were removed because of NaNs: {rows1 - rows2}")


industry.sort_values(by=["GeoFIPS", "GeoName"], inplace=True)

cols_to_save = industry.shape[1] - 2
industry = industry[["GeoFIPS", "GeoName"] + list(industry.columns[0:cols_to_save])]
industry = industry.reset_index(drop=True)

industry_pre2020 = industry[industry["Year"] < 2020]
industry_post2020 = industry[industry["Year"] >= 2020]


industry_list = [industry_pre2020, industry_post2020]

for i in range(len(industry_list)):
    industry_singl = industry_list[i]

    industry_singl = industry_singl.drop(columns=["Year"])

    columns_to_convert = industry_singl.columns[2:]
    industry_singl[columns_to_convert] = industry_singl[columns_to_convert].astype(
        float
    )

    industry_list[i] = industry_singl.reset_index(drop=True)


industry_pre2020, industry_post2020 = industry_list

for i in range(len(industry_list)):
    industry_singl = industry_list[i]

    row_sums = industry_singl.iloc[:, 3:].sum(axis=1)

    industry_singl.iloc[:, 3:] = industry_singl.iloc[:, 3:].div(row_sums, axis=0)
    industry_singl = industry_singl.drop(["employed_sum"], axis=1)

    industry_list[i] = industry_singl

industry_pre2020, industry_post2020 = industry_list

industry_pre2020.to_csv(f"{root}/data/raw/industry_pre2020_ct.csv", index=False)
industry_post2020.to_csv(f"{root}/data/raw/industry_post2020_ct.csv", index=False)
