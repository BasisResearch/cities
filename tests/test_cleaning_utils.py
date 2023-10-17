import os
import sys

import numpy as np
import pandas as pd

from cities.utils.cleaning_utils import standardize_and_scale

sys.path.insert(0, os.path.dirname(os.getcwd()))


# set up gdp data
gdp = pd.read_csv("../data/raw/CAGDP1_2001_2021.csv", encoding="ISO-8859-1")

gdp = gdp.drop(gdp.columns[2:8], axis=1)
gdp = gdp.drop("2012", axis=1)
gdp.replace("(NA)", np.nan, inplace=True)
gdp.replace("(NM)", np.nan, inplace=True)
gdp.dropna(axis=0, inplace=True)

for column in gdp.columns[2:]:
    gdp[column] = gdp[column].astype(float)
# set up ends


def test_standardize_and_scale():
    gdp_scaled = standardize_and_scale(gdp)

    for column in gdp_scaled.columns[2:]:
        assert np.min(gdp_scaled[column]) >= -1
        assert np.max(gdp_scaled[column]) <= 1

    assert gdp.shape == gdp_scaled.shape
