import os
#import pandas as pd
#from sqlalchemy import create_engine, MetaData

from cities.utils.data_grabber import find_repo_root

root = find_repo_root()



def list_csvs(csv_dir):

    csv_names = []
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            csv_names.append(filename)

    assert len(csv_names) > 10, f"Expected to find more than 10 csv files in {data_dir}, but found {len(csv_names)}"

    return csv_names