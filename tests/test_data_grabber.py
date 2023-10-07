import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))

import pytest
import pandas as pd
import numpy as np

from cities.utils import  DataGrabber


def test_DataGrabber():
    data = DataGrabber()

    data.get_gpd_wide()
    data.get_gpd_std_wide()
    data.get_gpd_long()
    data.get_gpd_std_long()
    assert data.gpd_wide.shape == (3079, 22)
    assert data.gpd_std_wide.shape == (3079, 22)
    assert data.gpd_long.shape == (61580, 4)
    assert data.gpd_std_long.shape == (61580, 4)



