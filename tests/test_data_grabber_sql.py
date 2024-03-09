import os

import pandas as pd
import pytest
from sqlalchemy import MetaData, create_engine

from cities.utils.data_grabber import find_repo_root
from cities.utils.sql_data_grabber import list_csvs

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
