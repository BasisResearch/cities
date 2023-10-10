import os

import pytest
import pandas as pd
import numpy as np

from cities.utils.data_grabber import  DataGrabber


def test_DataGrabber():
    data = DataGrabber()

    data.get_gdp_wide()
    data.get_gdp_std_wide()
    data.get_gdp_long()
    data.get_gdp_std_long()
    assert data.gdp_wide.shape[0] > 100
    assert data.gdp_std_wide.shape[1]  < 100
    assert data.gdp_long.shape[0] > 10000
    assert data.gdp_std_long.shape[1] == 4


    os.chdir(os.path.dirname(os.getcwd()))
    data2 = DataGrabber()

    data2.get_gdp_wide()
    data2.get_gdp_std_wide()
    data2.get_gdp_long()
    data2.get_gdp_std_long()
    assert data2.gdp_wide.shape[0] >100 
    assert data2.gdp_std_wide.shape[0] > 100
    assert data2.gdp_long.shape[0] > 10000
    assert data2.gdp_std_long.shape[1] == 4


    assert data.gdp_wide.equals(data2.gdp_wide)