import os
from pathlib import Path

from cities.utils.clean_variable import clean_variable

path = Path(__file__).parent.absolute()


def clean_unemployment():
    variable_name = "unemployment_rate"
    path_to_raw_csv = os.path.join(
        path, "../../data/raw/unemployment_rate_wide_withNA.csv"
    )
    clean_variable(variable_name, path_to_raw_csv)
