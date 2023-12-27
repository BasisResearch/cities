import pandas as pd
import numpy as np
from cities.utils.clean_variable import VariableCleanerMSA
from cities.utils.cleaning_utils import find_repo_root
root = find_repo_root()


def clean_industry_ma():

    cleaner = VariableCleanerMSA(variable_name="industry_compostion",
                              path_to_raw_csv = f"{root}/data/raw/industry_ma.csv")
    cleaner.clean_variable()

