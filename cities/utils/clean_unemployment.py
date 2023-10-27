import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cities.utils.clean_gdp import clean_gdp
from cities.utils.clean_variable import clean_variable
from cities.utils.cleaning_utils import standardize_and_scale
from cities.utils.data_grabber import DataGrabber

path = Path(__file__).parent.absolute()


def clean_unemployment():
    variable_name = "unemployment_rate"
    path_to_raw_csv = os.path.join(
        path, "../../data/raw/unemployment_rate_wide_withNA.csv"
    )
    clean_variable(variable_name, path_to_raw_csv)
