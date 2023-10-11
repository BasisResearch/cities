
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
    
    print("original length", original_length)
    # this assumes input df has two columns of metadata, then the rest are features
    # obey this convention with other datasets!
    my_id = df[df['GeoFIPS'] == fips].values[0][:2].copy()
    my_values = df[df['GeoFIPS'] == fips].values[0][2+lag:].copy()
    
    print("my_id", my_id)
    print("my_values", my_values)

    
    my_df = pd.concat([pd.DataFrame(my_id).T, pd.DataFrame(my_values).T], axis=1)

    print(my_df)
    
    assert fips in df['GeoFIPS'].values, "FIPS not found in the dataframe"
    other_df = df[df['GeoFIPS'] != fips].copy()
    print("other_df", other_df)
          
    my_array = np.array(my_values)
    
    print("my_array", my_array)
    
    other_df_cut = other_df.iloc[:, 2:-lag]
    print("other_cut", other_df_cut)
    other_arrays = np.array(other_df_cut)    
    
    print("other_arrays", other_arrays) 
  
    assert other_arrays.shape[0] + 1 == original_length, "Dataset sizes don't match"
    print(my_array.shape)
    print(other_arrays.shape)
    
    assert other_arrays.shape[1] == my_array.shape[0], "Lengths don't match"
    
    return {'my_array': my_array, 'other_arrays': other_arrays, "my_df": my_df,
            'other_df': other_df}
