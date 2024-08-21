import numpy as np
import pandas as pd
import requests
from us import states

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


variables = (
    "NAME,"
    "DP05_0070E,DP05_0072E,DP05_0073E,DP05_0074E,"
    "DP05_0075E,DP05_0077E,DP05_0078E,DP05_0079E,"
    "DP05_0080E,DP05_0081E,DP05_0082E,DP05_0083E"
)


county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html
year = 2019

dfs = []

for x in range(
    0, len(states.STATES)
):  # in this call it's not possible to use the '*' wildcard
    # to access all states, so we need to iterate over all states
    fips = states.STATES[x].fips

    url = (
        f"https://api.census.gov/data/{year}/acs/acs5/profile?"
        f"get={variables}&"
        f"for=tract:{tract}&"
        f"in=state:{fips}&"
        f"in=county:{county_fips}&"
        f"key={api_key}"
    )

    response = requests.get(url)

    assert (
        response.status_code == 200
    ), "The data retrieval went wrong"  # 200 means success

    print(f"{fips} fips done")

    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    dfs.append(df)

    combined_df_2019 = pd.concat(dfs, ignore_index=True)


# Different variable codes! As the definitions changed

variables = (
    "NAME,"
    "DP05_0072E,DP05_0074E,DP05_0075E,DP05_0076E,"
    "DP05_0077E,DP05_0079E,DP05_0080E,DP05_0081E,"
    "DP05_0082E,DP05_0083E,DP05_0084E,DP05_0085E"
)

county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html
year = 2022

dfs = []

for x in range(
    0, len(states.STATES)
):  # in this call it's not possible to use the '*' wildcard to
    # access all states, so we need to iterate over all states
    fips = states.STATES[x].fips

    url = (
        f"https://api.census.gov/data/{year}/acs/acs5/profile?"
        f"get={variables}&"
        f"for=tract:{tract}&"
        f"in=state:{fips}&"
        f"in=county:{county_fips}&"
        f"key={api_key}"
    )

    response = requests.get(url)

    assert (
        response.status_code == 200
    ), "The data retrieval went wrong"  # 200 means success

    print(f"{fips} fips done")

    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])

    dfs.append(df)

    combined_df_2022 = pd.concat(dfs, ignore_index=True)


ethnic_pre2020 = combined_df_2019.copy()
ethnic_post2020 = combined_df_2022.copy()

column_mapping_2019 = {
    "DP05_0070E": "total_pop",
    "DP05_0072E": "mexican",
    "DP05_0073E": "puerto_rican",
    "DP05_0074E": "cuban",
    "DP05_0075E": "other_hispanic_latino",
    "DP05_0077E": "white",
    "DP05_0078E": "black_african_american",
    "DP05_0079E": "american_indian_alaska_native",
    "DP05_0080E": "asian",
    "DP05_0081E": "native_hawaiian_other_pacific_islander",
    "DP05_0082E": "other_race",
    "DP05_0083E": "two_or_more_sum",
}


column_mapping_2022 = {
    "DP05_0072E": "total_pop",
    # those variable names work for 2022, be aware that in other years their meaning may differ
    "DP05_0074E": "mexican",
    "DP05_0075E": "puerto_rican",
    "DP05_0076E": "cuban",
    "DP05_0077E": "other_hispanic_latino",
    "DP05_0079E": "white",
    "DP05_0080E": "black_african_american",
    "DP05_0081E": "american_indian_alaska_native",
    "DP05_0082E": "asian",
    "DP05_0083E": "native_hawaiian_other_pacific_islander",
    "DP05_0084E": "other_race",
    "DP05_0085E": "two_or_more_sum",
}


ethnic_pre2020.rename(columns=column_mapping_2019, inplace=True)
ethnic_post2020.rename(columns=column_mapping_2022, inplace=True)


ethnic_list = [ethnic_pre2020, ethnic_post2020]

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


for i in range(len(ethnic_list)):
    ethnic = ethnic_list[i]

    ethnic["GeoFIPS"] = ethnic.apply(
        lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1
    ).astype(np.int64)

    ethnic.drop(["state", "county", "tract"], axis=1, inplace=True)

    ethnic["GeoName"] = ethnic["NAME"].apply(parse_geo_name).astype(str)

    assert (
        ethnic[ethnic["GeoName"] == "Unknown"].shape[0] == 0
    ), "There are Unknown GeoNames"

    ethnic.drop(["NAME"], axis=1, inplace=True)

    print(ethnic["GeoName"].nunique())

    ethnic_list[i] = ethnic

ethnic_pre2020, ethnic_post2020 = ethnic_list


ethnic_list = [ethnic_pre2020, ethnic_post2020]


for i in range(len(ethnic_list)):
    ethnic = ethnic_list[i]

    rows1 = ethnic.shape[0]
    ethnic.dropna(how="any", inplace=True)  # Drop NaN values inplace
    rows2 = ethnic.shape[0]
    print(f"This many rows were removed because of NaNs: {rows1 - rows2}")

    ethnic.sort_values(by=["GeoFIPS", "GeoName"], inplace=True)

    cols_to_save = ethnic.shape[1] - 2
    ethnic = ethnic[["GeoFIPS", "GeoName"] + list(ethnic.columns[0:cols_to_save])]

    ethnic_list[i] = ethnic.reset_index(drop=True)

ethnic_pre2020, ethnic_post2020 = ethnic_list


for i in range(len(ethnic_list)):
    ethnic = ethnic_list[i]

    ethnic.iloc[:, 2:] = ethnic.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    ethnic[ethnic.columns[2:]] = ethnic[ethnic.columns[2:]].astype(float)

    ethnic["other_race_races"] = ethnic["other_race"] + ethnic["two_or_more_sum"]

    ethnic = ethnic.drop(["other_race", "two_or_more_sum"], axis=1)

    ethnic["totalALT"] = ethnic.iloc[:, 3:].sum(axis=1)
    assert (ethnic["totalALT"] == ethnic["total_pop"]).all()

    ethnic = ethnic.drop("totalALT", axis=1)

    row_sums = ethnic.iloc[:, 2:].sum(axis=1)
    ethnic.iloc[:, 3:] = ethnic.iloc[:, 3:].div(row_sums, axis=0)

    ethnic = ethnic.drop(["total_pop"], axis=1)

    ethnic_list[i] = ethnic

ethnic_pre2020, ethnic_post2020 = ethnic_list


ethnic_pre2020.to_csv(
    f"{root}/data/raw/ethnic_composition_pre2020_filtered_wide.csv", index=False
)
ethnic_post2020.to_csv(
    f"{root}/data/raw/ethnic_composition_post2020_filtered_wide.csv", index=False
)
