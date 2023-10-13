import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  
sys.path.insert(0, parent_dir)


import pandas as pd
import numpy as np

import plotly.graph_objects as go
from scipy.spatial import distance
from cities.utils.data_grabber import DataGrabber
from cities.utils.similarity_utils import slice_with_lag



def test_slice_with_lag():
    
    df = pd.DataFrame()
    df['GeoFIPS'] = [1, 2, 3, 4, 5]
    df['GeoName'] = ['a', 'b', 'c', 'd', 'e']
    df['2001'] = [2001] * 5
    df['2002'] = [2002] * 5
    df['2003'] = [2003] * 5
    df['2004'] = [2004] * 5
    df['2005'] = [2005] * 5
    
    sliced = slice_with_lag(df, fips = 3, lag = 2)
    
    expected = np.array([[2001, 2002, 2003], [2001, 2002, 2003], [2001, 2002, 2003], [2001, 2002, 2003]])

    assert (sliced['my_array'] == [2003, 2004, 2005]).all()
    assert np.array_equal(sliced['other_arrays'], expected)
    assert sliced['other_df'].shape[0] == df.shape[0] -1
    

