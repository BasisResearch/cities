import os
import time

import sqlalchemy

from cities.utils.data_loader import select_from_sql

# local torch loader is needed for subsampling in evaluation, comparison to the previous dataset and useful for EDA

USERNAME = os.getenv("USERNAME")
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")

#####################
# data load and prep
#####################

kwargs = {
    "categorical": ["year", "census_tract"],
    "continuous": {
        "housing_units",
        "total_value",
        "median_value",
        "mean_limit_original",
        "median_distance",
        "income",
        "segregation_original",
        "white_original",
    },
    "outcome": "housing_units",
}

load_start = time.time()
with sqlalchemy.create_engine(
    f"postgresql://{USERNAME}@{HOST}/{DATABASE}"
).connect() as conn:
    subset = select_from_sql(
        "select * from dev.tracts_model__census_tracts order by census_tract, year",
        conn,
        kwargs,
    )
load_end = time.time()
print(f"Data loaded in {load_end - load_start} seconds")
