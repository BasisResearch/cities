import os

import pandas as pd
import pytest
from sqlalchemy import MetaData, create_engine

from cities.utils.data_grabber import (
    DataGrabberCSV,
    DataGrabberDB,
    MSADataGrabberCSV,
    find_repo_root,
    list_available_features,
    list_csvs,
)

smoke_test = "CI" in os.environ

if smoke_test:
    pytest.skip("Skipping all tests in this file during smoke tests", allow_module_level=True)


root = find_repo_root()

data_dirs = {
    "counties": os.path.join(root, "data/processed"),
    "msa": os.path.join(root, "data/MSA_level"),
}

database_paths = {
    "counties": os.path.join(root, "data/sql/counties_database.db"),
    "msa": os.path.join(root, "data/sql/msa_database.db"),
}


@pytest.mark.parametrize("level", ["counties", "msa"])
def test_database_tables(level):
    data_dir = data_dirs[level]
    database_path = database_paths[level]

    engine = create_engine(f"sqlite:///{database_path}", echo=True)
    csv_list = list_csvs(data_dir)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table_names = metadata.tables.keys()

    for csv in csv_list:
        assert csv[:-4] in table_names
        df_from_sql = pd.read_sql_table(csv[:-4], con=engine)
        df_from_csv = pd.read_csv(os.path.join(data_dir, csv))
        assert df_from_sql.equals(df_from_csv)
    engine.dispose()


@pytest.mark.parametrize("level", ["counties", "msa"])
def test_data_grabber(level):
    features = {
        "counties": list_available_features(),
        "msa": list_available_features("msa"),
    }[level]

    data_grabber_db = DataGrabberDB(level)
    data_grabber_csv = DataGrabberCSV() if level == "counties" else MSADataGrabberCSV()

    data_grabber_db.get_features_wide(features)
    data_grabber_csv.get_features_wide(features)

    data_grabber_db.get_features_std_wide(features)
    data_grabber_csv.get_features_std_wide(features)

    data_grabber_db.get_features_long(features)
    data_grabber_csv.get_features_long(features)

    data_grabber_db.get_features_std_long(features)
    data_grabber_csv.get_features_std_long(features)

    for feature in features:
        assert data_grabber_db.wide[feature].equals(data_grabber_csv.wide[feature])
        assert data_grabber_db.std_wide[feature].equals(
            data_grabber_csv.std_wide[feature]
        )
        assert data_grabber_db.long[feature].equals(data_grabber_csv.long[feature])
        assert data_grabber_db.std_long[feature].equals(
            data_grabber_csv.std_long[feature]
        )
