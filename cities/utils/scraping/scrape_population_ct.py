import numpy as np
import pandas as pd
import requests
from us import states

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()

# note that there is also a library for accessing the Census data:
# https://github.com/datamade/census


# description of the missing values in the data (it is sometimes caused by a mistake that can be corrected):
# https://www2.census.gov/geo/pdfs/reference/Geography_Notes.pdf


# THE PROCESS TAKES UP TO 15 MIN TO RUN ##############

# age variables, firstly let's focus just on the total population
# ,S0101_C01_002E,S0101_C01_003E,S0101_C01_004E,S0101_C01_005E,S0101_C01_006E,S0101_C01_007E,S0101_C01_008E,S0101_C01_009E,S0101_C01_010E,S0101_C01_011E,S0101_C01_012E,S0101_C01_013E,S0101_C01_014E,S0101_C01_015E,S0101_C01_016E,S0101_C01_017E,S0101_C01_018E,S0101_C01_019E

variables = "NAME,S0101_C01_001E"
county_fips = "*"  # all counties
tract = "*"  # all tracts
api_key = "077d857d6c12d5b9b3aeafa07d2c1916ba12a86c"
# private api key required to access the data https://api.census.gov/data/key_signup.html


interval = list(range(2010, 2023))  # Time series for ACS 5-year estimates

dfs = []


for year in interval:
    for x in range(
        0, len(states.STATES)
    ):  # in this call it's not possible to use the '*' wildcard to access all states, so we need to iterate over all states
        fips = states.STATES[x].fips

        url = (
            f"https://api.census.gov/data/{year}/acs/acs5/subject?"
            f"get={variables}&for=tract:{tract}&in=state:{fips}&"
            f"in=county:{county_fips}&key={api_key}"
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

pop = combined_df.copy()

column_mapping = {"S0101_C01_001E": "Value"}

pop.rename(columns=column_mapping, inplace=True)


# creating columns: GeoName, GeoFIPS

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

pop["GeoFIPS"] = pop.apply(
    lambda row: f"{row['state']}{row['county']}{row['tract']}", axis=1
).astype(np.int64)

pop.drop(["state", "county", "tract"], axis=1, inplace=True)


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


pop["GeoName"] = pop["NAME"].apply(parse_geo_name).astype(str)

assert pop[pop["GeoName"] == "Unknown"].shape[0] == 0, "There are Unknown GeoNames"

pop = pop.drop(["NAME"], axis=1)


# sorting

pop.sort_values(by=["Year", "GeoFIPS", "GeoName"], inplace=True)
pop = pop[["GeoFIPS", "GeoName", "Year", "Value"]].reset_index(drop=True)


# dividing the dataset into pre and post 2020

pop_pre2020 = pop[pop["Year"] < 2020].reset_index(drop=True)
pop_post2020 = pop[pop["Year"] >= 2020].reset_index(drop=True)

# deleting the rows with missing values

geo_counts = pop_pre2020["GeoFIPS"].value_counts()
geo_in_all_years = geo_counts[geo_counts == geo_counts.max()].index.tolist()
pop_pre2020_filtered = pop_pre2020[pop_pre2020["GeoFIPS"].isin(geo_in_all_years)]
missin_count = (
    pop_pre2020["GeoFIPS"].nunique() - pop_pre2020_filtered["GeoFIPS"].nunique()
)

print(f" {missin_count} GeoFIPS values were removed from the pre-2020 data")


geo_counts = pop_post2020["GeoFIPS"].value_counts()
geo_in_all_years = geo_counts[geo_counts == geo_counts.max()].index.tolist()
pop_post2020_filtered = pop_post2020[pop_post2020["GeoFIPS"].isin(geo_in_all_years)]
missin_count = (
    pop_post2020["GeoFIPS"].nunique() - pop_post2020_filtered["GeoFIPS"].nunique()
)

print(f" {missin_count} GeoFIPS values were removed from the post-2020 data")


# wide format

pop_pre2020_filtered_wide = pop_pre2020_filtered.pivot(
    index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
)
pop_pre2020_filtered_wide = pop_pre2020_filtered_wide.reset_index()
pop_pre2020_filtered_wide.columns.name = None


pop_post2020_filtered_wide = pop_post2020_filtered.pivot(
    index=["GeoFIPS", "GeoName"], columns="Year", values="Value"
)
pop_post2020_filtered_wide = pop_post2020_filtered_wide.reset_index()
pop_post2020_filtered_wide.columns.name = None


# sanity dropping of NA values

pop_pre2020_filtered_wide = pop_pre2020_filtered_wide.dropna(how="any")
pop_post2020_filtered_wide = pop_post2020_filtered_wide.dropna(how="any")

# sanity conversion to float64

columns_to_convert = pop_pre2020_filtered_wide.columns[2:]
pop_pre2020_filtered_wide[columns_to_convert] = pop_pre2020_filtered_wide[
    columns_to_convert
].astype(float)

columns_to_convert = pop_post2020_filtered_wide.columns[2:]
pop_post2020_filtered_wide[columns_to_convert] = pop_post2020_filtered_wide[
    columns_to_convert
].astype(float)

# saving raw data

pop_pre2020_filtered_wide.to_csv(
    f"{root}/data/raw/pop_pre2020_filtered_wide.csv", index=False
)
pop_post2020_filtered_wide.to_csv(
    f"{root}/data/raw/pop_post2020_filtered.csv", index=False
)
