import os

import pytest
import pandas as pd
import numpy as np
from cities.utils.data_grabber import  DataGrabber


features = ["gdp", "population"]

def test_DataGrabber():
    data = DataGrabber()

    data.get_features_wide(features)
    data.get_features_std_wide(features)
    data.get_features_long(features)
    data.get_features_std_long(features)
    
    for feature in features:
        assert data.wide[feature].shape[0] > 100
        assert data.std_wide[feature].shape[1]  < 100
        assert data.long[feature].shape[0] > 10000
        assert data.std_long[feature].shape[1] == 4


    os.chdir(os.path.dirname(os.getcwd()))
    data2 = DataGrabber()

    data2.get_features_wide(features)
    data2.get_features_std_wide(features)
    data2.get_features_long(features)
    data2.get_features_std_long(features)
    
    for feature in features:
        assert data2.wide[feature].shape[0] > 100
        assert data2.std_wide[feature].shape[1]  < 100
        assert data2.long[feature].shape[0] > 10000
        assert data2.std_long[feature].shape[1] == 4

    assert all(data.wide[feature].equals(data2.wide[feature]) for feature in features)

    
#    test_DataGrabber()