import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine


def find_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


root = find_repo_root()


def check_if_tensed(df):
    years_to_check = ["2015", "2018", "2019", "2020"] 
    check = any(year in df.columns for year in years_to_check)
    return check


class DataGrabberCSV:
    def __init__(self):
        self.repo_root = find_repo_root()
        self.data_path = os.path.join(self.repo_root, "data/processed")
        self.wide: Dict[str, pd.DataFrame] = {}
        self.std_wide: Dict[str, pd.DataFrame] = {}
        self.long: Dict[str, pd.DataFrame] = {}
        self.std_long: Dict[str, pd.DataFrame] = {}

    def _get_features(self, features: List[str], table_suffix: str) -> None:
        for feature in features:
            file_path = os.path.join(self.data_path, f"{feature}_{table_suffix}.csv")
            df = pd.read_csv(file_path)
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


class MSADataGrabberCSV(DataGrabberCSV):
    def __init__(self):
        super().__init__()
        self.repo_root = find_repo_root()
        self.data_path = os.path.join(self.repo_root, "data/MSA_level")
        sys.path.insert(0, self.data_path)

class CTDataGrabberCSV(DataGrabberCSV):
    def __init__(self,
                 level_DG: str = "pre_2020"):                 # new argument pre_2020 and post_2020
        super().__init__()
        self.data_path = os.path.join(self.repo_root, "data/Census_tract_level")
        self.level_DG = level_DG
        sys.path.insert(0, self.data_path)

    def _get_features(self, features: List[str], table_suffix: str) -> None:  # redefining data grabbing to depend on `level_DG` argument
        for feature in features:
            if self.level_DG == "pre_2020":
                file_path = os.path.join(self.data_path, f"{feature}_pre2020_CT_{table_suffix}.csv")
            elif self.level_DG == "post_2020":
                file_path = os.path.join(self.data_path, f"{feature}_post2020_CT_{table_suffix}.csv")
            else:
                raise ValueError("Invalid level_DG. Please choose 'pre_2020' or 'post_2020'.")

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
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
            else:
                print(f"File not found: {file_path}")

    def get_features_wide(self, features: List[str]) -> None:
        self._get_features(features, "wide")

    def get_features_std_wide(self, features: List[str]) -> None:
        self._get_features(features, "std_wide")

    def get_features_long(self, features: List[str]) -> None:
        self._get_features(features, "long")

    def get_features_std_long(self, features: List[str]) -> None:
        self._get_features(features, "std_long")



def list_available_features(level="county", level_DG="pre_2020"):
    root = find_repo_root()

    if level == "county":
        folder_path = f"{root}/data/processed"
    elif level == "msa":
        folder_path = f"{root}/data/MSA_level"
    elif level == "census_tract":
        folder_path = f"{root}/data/Census_tract_level"
    else:
        raise ValueError("Invalid level. Please choose 'county', 'census_tract' or 'msa'.")

    file_names = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
    processed_file_names = []

    for file_name in file_names:
        if level == "census_tract":
            if level_DG == "pre_2020" and "pre2020" in file_name:
                matches = re.split(r"_wide|_long|_std|_pre2020", file_name)
            elif level_DG == "post_2020" and "pre2020" not in file_name:
                matches = re.split(r"_wide|_long|_std|_post2020", file_name)
            else:
                continue
        else:
            matches = re.split(r"_wide|_long|_std", file_name)

        if matches:
            base_name = matches[0]
            processed_file_names.append(base_name)



    # Remove any remaining suffixes from the base names
    feature_names = list(set(processed_file_names))
    feature_names = [re.sub(r'(_pre2020|_post2020)$', '', name) for name in feature_names]

    return sorted(feature_names)


def list_tensed_features(level="county", level_DG="pre_2020"):
    if level == "county":
        data = DataGrabber()
        all_features = list_available_features(level="county")

    elif level == "msa":
        data = MSADataGrabber()
        all_features = list_available_features(level="msa")

    elif level == "census_tract":
        data = CTDataGrabberCSV(level_DG=level_DG)  # TODO: Change to CTDataGrabber() in the future
        all_features = list_available_features(level="census_tract", level_DG=level_DG)

    else:
        raise ValueError(
            "Invalid level. Please choose 'county', 'census_tract' or 'msa'."
        )

    data.get_features_wide(all_features)

    tensed_features = []
    for feature in all_features:
        if check_if_tensed(data.wide[feature]):
            tensed_features.append(feature)

    return sorted(tensed_features)


# TODO this only will pick up spending-based interventions
# needs to be modified/expanded when we add other types of interventions
def list_interventions():
    interventions = [
        feature for feature in list_tensed_features() if feature.startswith("spending_")
    ]
    return sorted(interventions)


def list_outcomes():
    outcomes = [
        feature
        for feature in list_tensed_features()
        if feature not in list_interventions()
    ]
    return sorted(outcomes)


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
        self.repo_root = find_repo_root()
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


DataGrabber = DataGrabberDB


def MSADataGrabberFactory():
    return DataGrabberDB(level="msa")


MSADataGrabber = MSADataGrabberFactory

# this reverts to csvs
# DataGrabber = DataGrabberCSV
# MSADataGrabber = MSADataGrabberCSV
