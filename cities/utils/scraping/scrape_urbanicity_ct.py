import numpy as np
import pandas as pd
import requests
from us import states

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()


# housing: total / urban / rural
# H2_001N,H2_002N,H2_003N"


# population: urban/rural/total
# P2_002N / P2_003N / P2_001N


variables = "NAME,H2_003N,H2_002N,H2_001N,P2_003N,P2_002N,P2_001N"
county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html
year = 2020


dfs = []


for x in range(
    0, len(states.STATES)
):  # in this call it's not possible to use the '*' wildcard to access all states, so we need to iterate over all states
    fips = states.STATES[x].fips

    url = (
        f"https://api.census.gov/data/{year}/dec/dhc?"
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
    df["Year"] = year  # Add the year column

    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


# housing: total / urban / rural
# H002001,H002002,H002005"

# population: rural/urban/total
# P002005/P002002/P002001


variables = (
    "NAME,H002005,H002002,H002001,P002005,P002002,P002001"  # different variable names
)
county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html
year = 2010


dfs = []


for x in range(
    0, len(states.STATES)
):  # in this call it's not possible to use the '*' wildcard to access all states, so we need to iterate over all states
    fips = states.STATES[x].fips

    url = (
        f"https://api.census.gov/data/{year}/dec/sf1?"
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
    df["Year"] = year  # Add the year column

    dfs.append(df)

combined_df_pre2020 = pd.concat(dfs, ignore_index=True)


urbanicity_post2020 = combined_df.copy()
urbanicity_pre2020 = combined_df_pre2020.copy()


columns_pre2020 = {
    "H002001": "total_housing",
    "H002002": "urban_housing",
    "H002005": "rural_housing",
    "P002001": "total_pop",
    "P002002": "urban_pop",
    "P002005": "rural_pop",
}

columns_post2020 = {
    "H2_001N": "total_housing",
    "H2_002N": "urban_housing",
    "H2_003N": "rural_housing",
    "P2_001N": "total_pop",
    "P2_002N": "urban_pop",
    "P2_003N": "rural_pop",
}

urbanicity_pre2020.rename(columns=columns_pre2020, inplace=True)
urbanicity_post2020.rename(columns=columns_post2020, inplace=True)

urban_list = [urbanicity_pre2020, urbanicity_post2020]

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


for i in range(len(urban_list)):
    urban_singl = urban_list[i]

    urban_singl["GeoFIPS"] = urban_singl.apply(
        lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1
    ).astype(np.int64)

    urban_singl.drop(["state", "county", "tract"], axis=1, inplace=True)

    urban_singl["GeoName"] = urban_singl["NAME"].apply(parse_geo_name).astype(str)

    assert (
        urban_singl[urban_singl["GeoName"] == "Unknown"].shape[0] == 0
    ), "There are Unknown GeoNames"

    urban_singl.drop(["NAME"], axis=1, inplace=True)

    print(urban_singl["GeoName"].nunique())

    urban_list[i] = urban_singl

urbanicity_pre2020, urbanicity_post2020 = urban_list

urban_list = [urbanicity_pre2020, urbanicity_post2020]


for i in range(len(urban_list)):
    urban_singl = urban_list[i]

    rows1 = urban_singl.shape[0]
    urban_singl.dropna(how="any", inplace=True)  # Drop NaN values inplace
    rows2 = urban_singl.shape[0]
    print(f"This many rows were removed because of NaNs: {rows1 - rows2}")

    if "Year" in urban_singl.columns:
        urban_singl.drop(columns=["Year"], inplace=True)

    urban_singl.sort_values(by=["GeoFIPS", "GeoName"], inplace=True)

    cols_to_save = urban_singl.shape[1] - 2
    urban_singl = urban_singl[
        ["GeoFIPS", "GeoName"] + list(urban_singl.columns[0:cols_to_save])
    ]

    urban_list[i] = urban_singl.reset_index(drop=True)

urbanicity_pre2020, urbanicity_post2020 = urban_list

for i in range(len(urban_list)):
    urban_singl = urban_list[i]

    columns_to_convert = urban_singl.columns[2:]
    urban_singl[columns_to_convert] = urban_singl[columns_to_convert].astype(float)

    urban_list[i] = urban_singl.reset_index(drop=True)


urbanicity_pre2020, urbanicity_post2020 = urban_list

for i in range(len(urban_list)):
    urban_singl = urban_list[i]

    urban_singl["rural_pop_prct"] = urban_singl["rural_pop"] / urban_singl["total_pop"]
    urban_singl["rural_housing_prct"] = (
        urban_singl["rural_housing"] / urban_singl["total_housing"]
    )

    urban_singl.drop(columns=["total_pop", "total_housing"], axis=1, inplace=True)

    columns_to_convert = urban_singl.columns[2:]
    urban_singl[columns_to_convert] = urban_singl[columns_to_convert].astype(float)

    urban_singl.reset_index(drop=True, inplace=True)

    urban_list[i] = urban_singl

urbanicity_pre2020, urbanicity_post2020 = urban_list

urbanicity_pre2020.to_csv(f"{root}/data/raw/urbanicity_pre2020_ct.csv", index=False)
urbanicity_post2020.to_csv(f"{root}/data/raw/urbanicity_post2020_ct.csv", index=False)
