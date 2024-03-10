import os
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine

from cities.utils.data_grabber import find_repo_root

# import pandas as pd
# from sqlalchemy import create_engine, MetaData


root = find_repo_root()


def list_csvs(csv_dir):
    csv_names = []
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            csv_names.append(filename)

    assert (
        len(csv_names) > 10
    ), f"Expected to find more than 10 csv files in {csv_dir}, but found {len(csv_names)}"

    return csv_names


# these paths will need to be updated in deployment
# as is, this assumes you've run `csv_to_db_pipeline.py`
DB_PATHS = {
    "counties": os.path.join(root, "data/sql/counties_database.db"),
    "msa": os.path.join(root, "data/sql/msa_database.db"),
}

if not os.path.exists(DB_PATHS["counties"]):
    raise FileNotFoundError(
        f"No db at {DB_PATHS['counties']}, run `csv_to_db_pipeline.py` first."
    )
if not os.path.exists(DB_PATHS["msa"]):
    raise FileNotFoundError(
        f"No db at {DB_PATHS['msa']}, run `csv_to_db_pipeline.py` first."
    )


# this check might need to be revised in deployment
if "COUNTIES_ENGINE" not in locals():
    counties_engine = create_engine(f'sqlite:///{DB_PATHS["counties"]}')

if "MSA_ENGINE" not in locals():
    msa_engine = create_engine(f'sqlite:///{DB_PATHS["msa"]}')

engines = {"counties": counties_engine, "msa": msa_engine}


class DataGrabberDB:
    def __init__(self, level: str = "counties"):
        self.engine = engines[level]
        self.wide: Dict[str, pd.DataFrame] = {}
        self.std_wide: Dict[str, pd.DataFrame] = {}
        self.long: Dict[str, pd.DataFrame] = {}
        self.std_long: Dict[str, pd.DataFrame] = {}

    def _get_features(self, features: List[str], table_suffix: str) -> None:
        for feature in features:
            table_name = f"{feature}_{table_suffix}"
            df = pd.read_sql_table(table_name, con=self.engine)
            if table_suffix == "wide":
                self.wide[feature] = df
            elif table_suffix == "std_wide":
                self.std_wide[feature] = df
            elif table_suffix == "long":
                self.long[feature] = df
            elif table_suffix == "std_long":
                self.std_long[feature] = df
            else:
                raise ValueError(
                    "Invalid table suffix. Please choose 'wide', 'std_wide', 'long', or 'std_long'."
                )

    def get_features_wide(self, features: List[str]) -> None:
        self._get_features(features, "wide")

    def get_features_std_wide(self, features: List[str]) -> None:
        self._get_features(features, "std_wide")

    def get_features_long(self, features: List[str]) -> None:
        self._get_features(features, "long")

    def get_features_std_long(self, features: List[str]) -> None:
        self._get_features(features, "std_long")


class MSADataGrabberDB(DataGrabberDB):
    def __init__(self):
        super().__init__()
        self.engine = msa_engine


# # Assuming you have already created the database engine
# engine = create_engine('sqlite:///path/to/your/database.db')

# # Instantiate DataGrabberDB with the database engine
# data_grabber = DataGrabberDB(engine)

# # Example usage
# features_to_get = ["feature1", "feature2"]
# data_grabber.get_features_wide(features_to_get)
# data_grabber.get_features_std_wide(features_to_get)
# data_grabber.get_features_long(features_to_get)
# data_grabber.get_features_std_long(features_to_get)
