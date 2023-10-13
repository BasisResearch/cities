
from typing import Dict, List, Union
import numpy as np
import pandas as pd


def slice_with_lag(df: pd.DataFrame, fips: int, lag: int) -> Dict[str, np.ndarray]:
    """
    Takes a pandas dataframe, a location FIPS and a lag (years),
    returns a dictionary with two numpy arrays:
    - my_array: the array of features for the location with the given FIPS
    - other_arrays: the array of features for all other locations
    if lag>0, drops first lag columns from my_array and last lag columns from other_arrays.
    Meant to be used prior to calculating similarity.
    """
    original_length = df.shape[0]
        
    # this assumes input df has two columns of metadata, then the rest are features
    # obey this convention with other datasets!
    my_array = np.array(df[df['GeoFIPS'] == fips].values[0][2+lag:].copy())
    other_df = df[df['GeoFIPS'] != fips].copy()
    
    if lag >0:
        other_df_cut = other_df.iloc[:, 2:-lag]
        other_arrays = np.array(other_df_cut.values)
    else:
        other_df_cut = other_df.iloc[:, 2:]
        other_arrays = np.array(other_df_cut.values)
            
    assert other_arrays.shape[0] + 1 == original_length, "Dataset sizes don't match"
    assert other_arrays.shape[1] == my_array.shape[0], "Lengths don't match"
    
    return {'my_array': my_array, 'other_arrays': other_arrays, 'other_df': other_df}
