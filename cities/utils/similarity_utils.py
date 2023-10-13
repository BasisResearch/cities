
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import math
from plotly import graph_objs as go


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
    original_array_width = df.shape[1] - 2

    #assert error if lag > original array width
    assert lag <= original_array_width, "Lag is greater than the number of years in the dataframe"
    assert lag >= 0, "Lag must be a positive integer" 

    # this assumes input df has two columns of metadata, then the rest are features
    # obey this convention with other datasets!
    
    my_row = df.loc[df['GeoFIPS'] == fips].copy()
    my_id = my_row[['GeoFIPS', 'GeoName']]
    my_values = my_row.iloc[:, 2 + lag:]

    my_df = pd.concat([my_id, my_values], axis=1)

    my_df = pd.DataFrame({**my_id.to_dict(orient='list'), **my_values.to_dict(orient='list')})

    assert fips in df['GeoFIPS'].values, "FIPS not found in the dataframe"
    other_df = df[df['GeoFIPS'] != fips].copy()
    
    my_array = np.array(my_values)
    
    if lag > 0:
        other_df = df[df['GeoFIPS'] != fips].iloc[:,:-lag]

    assert fips not in other_df['GeoFIPS'].values, "FIPS found in the other dataframe"
    other_arrays = np.array(other_df.iloc[:, 2:])   

    assert other_arrays.shape[0] + 1 == original_length, "Dataset sizes don't match"
    assert other_arrays.shape[1] == my_array.shape[1], "Lengths don't match"
    
    return {'my_array': my_array, 'other_arrays': other_arrays, "my_df": my_df,
            'other_df': other_df}


def compute_weight_array(object, rate = 1.08):

    def divide_exponentially(k, n, r):
        result = []
        denominator = sum([r ** j for j in range(n)])
        for i in range(n):
            value = k * (r ** i) / denominator
            result.append(value)
        return result

    def check_if_tensed(df):
        years_to_check = ['2015', '2018', '2019', '2020']
        check = df.columns[2:].isin(years_to_check).any().any()
        return check

    max_other_scores = sum(object.weights.values())

    weight_outcome_joint  = max_other_scores if max_other_scores > 0 else 1 
    object.weights[object.outcome_var] = weight_outcome_joint

    tensed_status = {}
    columns = {}
    column_counts = {}
    column_tags = []
    weight_lists = {}
    all_columns = []
    for feature in object.all_features:
        tensed_status[feature] = check_if_tensed(object.data.std_wide[feature])
        columns[feature] = object.data.std_wide[feature].columns[2:]
        column_counts[feature] = len(object.data.std_wide[feature].columns) - 2
        all_columns.extend([f"{column}_{feature}" for column in object.data.std_wide[feature].columns[2:]])
        column_tags.extend([feature] * column_counts[feature])
        if tensed_status[feature]:
            weight_lists[feature] = divide_exponentially(object.weights[feature], column_counts[feature], rate)
        else:
            weight_lists[feature] = [object.weights[feature] / column_counts[feature]] * column_counts[feature]
    
    all_weights = np.concatenate(list(weight_lists.values()))

    fig = go.Figure()

    fig.add_trace(go.Bar(x=all_columns, y=all_weights))

    fig.update_layout(
        xaxis_title="columns",
        yaxis_title="weights",
        title="Weights of columns",
        template = "plotly_white" 
    )

    object.weigth_plot = fig
    object.all_weights = all_weights

    return np.array(all_weights)
    
