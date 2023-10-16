import os

import numpy as np
import pandas as pd
import pytest

from cities.utils.data_grabber import DataGrabber

# for some reason this test succeeds only
# with pytest run as a module.
# python -m pytest test_data_grabber.py
# TODO fix this

features = ["gdp", "population", "transport"]


def test_DataGrabber():
    data = DataGrabber()

    data.get_features_wide(features)
    data.get_features_std_wide(features)
    data.get_features_long(features)
    data.get_features_std_long(features)

    for feature in features:
        assert data.wide[feature].shape[0] > 100
        assert data.std_wide[feature].shape[1] < 100
        assert data.long[feature].shape[0] > 1000
        assert data.std_long[feature].shape[1] == 4

    for feature in features:  # fine
        dataTypeError = "Wrong data type!"
        assert data.wide[feature].iloc[:, 0].dtype == int, dataTypeError
        assert data.wide[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.std_wide[feature].iloc[:, 0].dtype == int, dataTypeError
        assert data.std_wide[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.long[feature].iloc[:, 0].dtype == int, dataTypeError
        assert data.long[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.std_long[feature].iloc[:, 0].dtype == int, dataTypeError
        assert data.std_long[feature].iloc[:, 1].dtype == object, dataTypeError

    for feature in features:
        for column in data.wide[feature].columns[2:]:
            std_error = "Standarization error"
            assert (
                data.wide[feature][column].dtype == float
            ), f"The column '{column}' is not of float type."
            assert (
                data.std_wide[feature][column].dtype == float
            ), f"The column '{column}' is not of float type."
            assert (data.std_wide[feature][column] >= -1).all() and (
                data.std_wide[feature][column] <= 1
            ).all(), std_error

    for column in data.long[feature].columns[3:]:
        assert (
            data.long[feature][column].dtype == float
        ), f"The column '{column}' is not of float or int type."
        assert (
            data.std_long[feature][column].dtype == float
        ), f"The column '{column}' is not of float or int type."

    for feature in features:
        assert data.std_long[feature].iloc[:, 2].dtype in (
            float,
            int,
            object,
        ), f"The column '{column}' is not of float or int type."
        assert data.long[feature].iloc[:, 2].dtype in (
            float,
            int,
            object,
        ), f"The column '{column}' is not of float or int type."

    for feature in features:
        for column in data.std_long[feature].columns[3:]:
            assert (data.std_long[feature][column] >= -1).all() and (
                data.std_long[feature][column] <= 1
            ).all()

    os.chdir(os.path.dirname(os.getcwd()))
    data2 = DataGrabber()

    data2.get_features_wide(features)
    data2.get_features_std_wide(features)
    data2.get_features_long(features)
    data2.get_features_std_long(features)

    for feature in features:
        assert data2.wide[feature].shape[0] > 100
        assert data2.std_wide[feature].shape[1] < 100
        assert data2.long[feature].shape[0] > 1000
        assert data2.std_long[feature].shape[1] == 4

    assert all(data.wide[feature].equals(data2.wide[feature]) for feature in features)
