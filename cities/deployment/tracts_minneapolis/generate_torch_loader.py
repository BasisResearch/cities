import os
import time

import sqlalchemy
import torch
from dotenv import load_dotenv

from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import ZoningDataset, select_from_sql

load_dotenv()

local_user = os.getenv("USER")
if local_user == "rafal":
    load_dotenv(os.path.expanduser("~/.env_pw"))
# local torch loader is needed for subsampling in evaluation, comparison to the previous dataset and useful for ED
DB_USERNAME = os.getenv("DB_USERNAME")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
PASSWORD = os.getenv("PASSWORD")


#####################
# data load and prep
#####################

kwargs = {
    "categorical": ["year", "census_tract"],
    "continuous": {
        "housing_units",
        "housing_units_original",
        "total_value",
        "total_value_original",
        "median_value",
        "mean_limit_original",
        "median_distance",
        "income",
        "segregation_original",
        "white_original",
        "parcel_sqm",
    },
    "outcome": "housing_units",
}

load_start = time.time()
with sqlalchemy.create_engine(
    f"postgresql://{DB_USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
).connect() as conn:
    subset = select_from_sql(
        "select * from dev.tracts_model__census_tracts order by census_tract, year",
        conn,
        kwargs,
    )
load_end = time.time()
print(f"Data loaded in {load_end - load_start} seconds")


columns_to_standardize = [
    "housing_units_original",
    "total_value_original",
]

new_standardization_dict = {}

for column in columns_to_standardize:
    new_standardization_dict[column] = {
        "mean": subset["continuous"][column].mean(),
        "std": subset["continuous"][column].std(),
    }


assert "parcel_sqm" in subset["continuous"].keys()

root = find_repo_root()

pg_census_tracts_dataset = ZoningDataset(
    subset["categorical"],
    subset["continuous"],
    standardization_dictionary=new_standardization_dict,
)
assert "parcel_sqm" in subset["continuous"].keys()

pg_census_tracts_data_path = os.path.join(
    root, "data/minneapolis/processed/pg_census_tracts_dataset.pt"
)

torch.save(pg_census_tracts_dataset, pg_census_tracts_data_path)
