#from scipy.spatial import distance
#from cities.utils.data_grabber import DataGrabber
#import os
#print("current_environment", os.environ['CONDA_DEFAULT_ENV'])
from cities.utils.similarity_utils import (slice_with_lag, compute_weight_array)
from cities.utils.data_grabber import DataGrabber
from cities.queries.fips_query import FipsQuery

from scipy.spatial import distance
import matplotlib.pyplot as plt
#import sys
#print(sys.path)
#parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  
#sys.path.insert(0, parent_dir)



import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize("lag", [0, 2])
def test_slice_with_lag(lag):
    print("mylag", lag)

    df = pd.DataFrame()
    df['GeoFIPS'] = [0, 1, 2, 3, 4]
    df['GeoName'] = ['a', 'b', 'c', 'd', 'e']
    df['2001'] = [20010, 20011, 20012, 20013, 20014] 
    df['2002'] = [20020, 20021, 20022, 20023, 20024]
    df['2003'] = [20030, 20031, 20032, 20033, 20034]
    df['2004'] = [20040, 20041, 20042, 20043, 20044]
    df['2005'] = [20050, 20051, 20052, 20053, 20054]
    

    sliced = slice_with_lag(df, fips = 2, lag = lag)


    expected0 = np.array([[20010, 20020, 20030, 20040, 20050],
                           [20011, 20021, 20031, 20041, 20051],
                            [20013, 20023, 20033, 20043, 20053],
                            [20014, 20024, 20034, 20044, 20054]])
    
    expected2 = np.array([[20010, 20020, 20030], [20011, 20021, 20031],
                           [20013, 20023, 20033], [20014, 20024, 20034]])

    assert sliced['other_df'].shape[0] == df.shape[0] -1

    if lag == 0:
        assert (sliced['my_array'] == [20012, 20022, 20032, 20042, 20052]).all()
        assert np.array_equal(sliced['other_arrays'], expected0)

    if lag == 2:
        assert (sliced['my_array'] == [20032, 20042, 20052]).all()
        assert np.array_equal(sliced['other_arrays'], expected2)



def test_slice_with_lag_on_real_data():
    outcome_var = "gdp"
    fips = 21001
    lag = 0

    data = DataGrabber()
    data.get_features_std_wide([outcome_var])

    outcome_slices = slice_with_lag(data.std_wide[outcome_var],
                                               fips, lag)
 
    print(data.std_wide[outcome_var].shape)

    
    assert data.std_wide[outcome_var].duplicated().sum() == 0 
    

    my_array = outcome_slices['my_array']
    other_arrays = outcome_slices['other_arrays']

    distances = []
    for vector in other_arrays:
        distances.append(distance.euclidean(np.squeeze(my_array), vector))
        
    print(other_arrays.shape)

    count = sum(1 for distance in distances if distance == 0)
    print("Number of zeros:", count)

    assert count == 0


def test_compute_weight_array():
    f  = FipsQuery(42001, "gdp", lag = 0, top =5)
    
    fp = FipsQuery(42001, "gdp", feature_groups= ["population"],
                  weights= {"population": 4}, lag = 0, top =5)
    
    compute_weight_array(f)
    
    compute_weight_array(fp)

    assert len(f.all_weights) == f.data.std_wide['gdp'].shape[1] - 2
    assert len(fp.all_weights) == fp.data.std_wide['gdp'].shape[1] + fp.data.std_wide['population'].shape[1] - 4



