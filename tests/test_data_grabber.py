import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())
print(sys.path)
import pytest
import pandas as pd
import numpy as np

from cities.utils import  DataGrabber


def test_DataGrabber():
    data = DataGrabber()

    data.get_gdp_wide()
    data.get_gdp_std_wide()
    data.get_gdp_long()
    data.get_gdp_std_long()
    assert data.gdp_wide.shape == (3079, 22)
    assert data.gdp_std_wide.shape == (3079, 22)
    assert data.gdp_long.shape == (61580, 4)
    assert data.gdp_std_long.shape == (61580, 4)


    os.chdir(os.path.dirname(os.getcwd()))
    data2 = DataGrabber()

    data2.get_gdp_wide()
    data2.get_gdp_std_wide()
    data2.get_gdp_long()
    data2.get_gdp_std_long()
    assert data2.gdp_wide.shape == (3079, 22)
    assert data2.gdp_std_wide.shape == (3079, 22)
    assert data2.gdp_long.shape == (61580, 4)
    assert data2.gdp_std_long.shape == (61580, 4)

