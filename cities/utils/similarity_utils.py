from typing import Dict

import numpy as np
import pandas as pd
from plotly import graph_objs as go

from cities.utils.cleaning_utils import check_if_tensed


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

    # assert error if lag > original array width
    assert (
        lag <= original_array_width
    ), "Lag is greater than the number of years in the dataframe"
    assert lag >= 0, "Lag must be a positive integer"

    # this assumes input df has two columns of metadata, then the rest are features
    # obey this convention with other datasets!

    my_row = df.loc[df["GeoFIPS"] == fips].copy()
    my_id = my_row[["GeoFIPS", "GeoName"]]
    my_values = my_row.iloc[:, 2 + lag :]

    my_df = pd.concat([my_id, my_values], axis=1)

    my_df = pd.DataFrame(
        {**my_id.to_dict(orient="list"), **my_values.to_dict(orient="list")}
    )

    assert fips in df["GeoFIPS"].values, "FIPS not found in the dataframe"
    other_df = df[df["GeoFIPS"] != fips].copy()

    my_array = np.array(my_values)

    if lag > 0:
        other_df = df[df["GeoFIPS"] != fips].iloc[:, :-lag]

    assert fips not in other_df["GeoFIPS"].values, "FIPS found in the other dataframe"
    other_arrays = np.array(other_df.iloc[:, 2:])

    assert other_arrays.shape[0] + 1 == original_length, "Dataset sizes don't match"
    assert other_arrays.shape[1] == my_array.shape[1], "Lengths don't match"

    return {
        "my_array": my_array,
        "other_arrays": other_arrays,
        "my_df": my_df,
        "other_df": other_df,
    }


def generalized_euclidean_distance(u, v, weights):
    result = sum(
        abs(weights)
        * ((weights >= 0) * abs(u - v) + (weights < 0) * (-abs(u - v) + 2)) ** 2
    ) ** (1 / 2)
    return result


def divide_exponentially(group_weight, number_of_features, rate):
    """
    Returns a list of `number_of_features` weights that sum to `group_weight` and are distributed
    exponentially. Intended for time series feature groups.
    If `rate` is 1, all weights are equal. If `rate` is greater than 1, weights
    prefer more recent events.
    """
    result = []
    denominator = sum([rate**j for j in range(number_of_features)])
    for i in range(number_of_features):
        value = group_weight * (rate**i) / denominator
        result.append(value)
    return result


def compute_weight_array(query_object, rate=1.08):

    assert (
        sum(
            abs(value)
            for key, value in query_object.feature_groups_with_weights.items()
        )
        != 0
    ), "At least one weight has to be other than 0"

    max_other_scores = sum(
        abs(value)
        for key, value in query_object.feature_groups_with_weights.items()
        if key != query_object.outcome_var
    )

    if (
        query_object.outcome_var
        and query_object.feature_groups_with_weights[query_object.outcome_var] != 0
    ):
        weight_outcome_joint = max_other_scores if max_other_scores > 0 else 1
        query_object.feature_groups_with_weights[query_object.outcome_var] = (
            weight_outcome_joint
            * query_object.feature_groups_with_weights[query_object.outcome_var]
        )

    tensed_status = {}
    columns = {}
    column_counts = {}
    weight_lists = {}
    all_columns = []
    for feature in query_object.feature_groups:
        tensed_status[feature] = check_if_tensed(query_object.data.std_wide[feature])

        columns[feature] = query_object.data.std_wide[feature].columns[2:]

        column_counts[feature] = len(query_object.data.std_wide[feature].columns) - 2

        if feature == query_object.outcome_var and query_object.lag > 0:
            column_counts[feature] -= query_object.lag

        all_columns.extend(
            [
                f"{column}_{feature}"
                for column in query_object.data.std_wide[feature].columns[2:]
            ]
        )

        # TODO: remove if tests passed
        # column_tags.extend([feature] * column_counts[feature])
        if tensed_status[feature]:
            weight_lists[feature] = divide_exponentially(
                query_object.feature_groups_with_weights[feature],
                column_counts[feature],
                rate,
            )
        else:
            weight_lists[feature] = [
                query_object.feature_groups_with_weights[feature]
                / column_counts[feature]
            ] * column_counts[feature]

    query_object.all_columns = all_columns[query_object.lag :]
    query_object.all_weights = np.concatenate(list(weight_lists.values()))


def plot_weights(query_object):
    fig = go.Figure()

    fig.add_trace(go.Bar(x=query_object.all_columns, y=query_object.all_weights))

    fig.update_layout(
        xaxis_title="columns",
        yaxis_title="weights",
        title="Weights of columns",
        template="plotly_white",
    )

    query_object.weigth_plot = fig
    query_object.weigth_plot.show()
