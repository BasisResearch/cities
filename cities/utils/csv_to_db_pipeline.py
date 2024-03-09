import logging
import os
import time

import pandas as pd
from sqlalchemy import create_engine

from cities.utils.data_grabber import find_repo_root
from cities.utils.sql_data_grabber import list_csvs

logging.disable(logging.WARNING)


def create_database(data_dir, database_path):
    engine = create_engine(f"sqlite:///{database_path}", echo=True)
    csv_list = list_csvs(data_dir)

    if not os.path.exists(database_path):
        print("Database not found at", database_path)

    for csv in csv_list:
        df = pd.read_csv(os.path.join(data_dir, csv))
        df.to_sql(
            csv[:-4],
            con=engine,
            if_exists="replace",
            index=False,
            index_label="GeoFiPS",
        )

    engine.dispose()


def main():
    root = find_repo_root()
    levels = ["counties", "msa"]

    data_dirs = {
        "counties": os.path.join(root, "data/processed"),
        "msa": os.path.join(root, "data/MSA_level"),
    }

    database_paths = {
        "counties": os.path.join(root, "data/sql/counties_database.db"),
        "msa": os.path.join(root, "data/sql/msa_database.db"),
    }

    for level in levels:
        data_dir = data_dirs[level]
        database_path = database_paths[level]
        create_database(data_dir, database_path)


start = time.time()
if __name__ == "__main__":
    main()

print(
    "All csv tables are now included in the databases. Time elapsed:",
    time.time() - start,
)
