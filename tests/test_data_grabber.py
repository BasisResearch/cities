import os

import numpy as np
import pandas as pd

from cities.utils.data_grabber import (
    DataGrabber,
    MSADataGrabber,
    list_available_features,
    list_interventions,
    list_outcomes,
    list_tensed_features,
)

features = list_available_features()
features_msa = list_available_features("msa")


def test_non_emptiness_DataGrabber():
    assert features is not None


data = DataGrabber()

data.get_features_wide(features)
data.get_features_std_wide(features)
data.get_features_long(features)
data.get_features_std_long(features)

for feature in features:
    assert data.wide[feature].shape[0] > 2800
    assert data.std_wide[feature].shape[1] < 100
    assert data.long[feature].shape[0] > 2800
    assert data.std_long[feature].shape[1] == 4
    assert (
        data.wide["gdp"]["GeoFIPS"].nunique() == data.wide[feature]["GeoFIPS"].nunique()
    )
    assert (
        data.long["gdp"]["GeoFIPS"].nunique() == data.long[feature]["GeoFIPS"].nunique()
    )


def test_non_emptiness_MSADataGrabber():
    os.chdir(os.path.dirname(os.getcwd()))
    data_msa = MSADataGrabber()

    data_msa.get_features_wide(features_msa)
    data_msa.get_features_std_wide(features_msa)
    data_msa.get_features_long(features_msa)
    data_msa.get_features_std_long(features_msa)

    for feature in features_msa:
        assert data_msa.wide[feature].shape[0] > 100
        assert data_msa.std_wide[feature].shape[1] < 100
        assert data_msa.long[feature].shape[0] > 100
        assert data_msa.std_long[feature].shape[1] == 4


def general_data_format_testing(data, features):
    assert features is not None

    data.get_features_wide(features)
    data.get_features_std_wide(features)
    data.get_features_long(features)
    data.get_features_std_long(features)

    for feature in features:
        dataTypeError = "Wrong data type!"
        assert data.wide[feature].iloc[:, 0].dtype == np.int64, dataTypeError
        assert data.wide[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.std_wide[feature].iloc[:, 0].dtype == np.int64, dataTypeError
        assert data.std_wide[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.long[feature].iloc[:, 0].dtype == np.int64, dataTypeError
        assert data.long[feature].iloc[:, 1].dtype == object, dataTypeError
        assert data.std_long[feature].iloc[:, 0].dtype == np.int64, dataTypeError
        assert data.std_long[feature].iloc[:, 1].dtype == object, dataTypeError

    for feature in features:
        namesFipsError = "FIPS codes and GeoNames don't match!"
        assert (
            data.wide[feature]["GeoFIPS"].nunique()
            == data.wide[feature]["GeoName"].nunique()
        ), namesFipsError
        assert (
            data.long[feature]["GeoFIPS"].nunique()
            == data.long[feature]["GeoName"].nunique()
        ), namesFipsError
        assert (
            data.std_wide[feature]["GeoFIPS"].nunique()
            == data.std_wide[feature]["GeoName"].nunique()
        ), namesFipsError
        assert (
            data.std_long[feature]["GeoFIPS"].nunique()
            == data.std_long[feature]["GeoName"].nunique()
        ), namesFipsError

    for feature in features:
        for column in data.wide[feature].columns[2:]:
            std_error = f"Standarization error for {column} in {feature}"
            assert (
                data.wide[feature][column].dtype == float
            ), f"The column '{column}' of feature '{feature}' is not of float type."
            assert (
                data.std_wide[feature][column].dtype == float
            ), f"The column '{column}' is not of float type."
            assert (data.std_wide[feature][column] >= -1).all() and (
                data.std_wide[feature][column] <= 1
            ).all(), std_error

    for column in data.long[feature].columns[3:]:
        assert (
            data.long[feature][column].dtype == float
        ), f"The column '{column}' of feature '{feature}' is not of float or int type."
        assert (
            data.std_long[feature][column].dtype == float
        ), f"The column '{column}' of feature '{feature}' is not of float or int type."

    for feature in features:
        assert data.std_long[feature].iloc[:, 2].dtype in (
            float,
            np.int64,
            object,
        ), f"The column '{column}' of feature '{feature}' is not of float or int type."
        assert data.long[feature].iloc[:, 2].dtype in (
            float,
            np.int64,
            object,
        ), f"The column '{column}' of feature '{feature}' is not of float or int type."

    for feature in features:
        for column in data.std_long[feature].columns[3:]:
            assert (data.std_long[feature][column] >= -1).all() and (
                data.std_long[feature][column] <= 1
            ).all(), (
                f"The column '{column}' of feature '{feature}' is not standardized."
            )


def check_years(df):
    current_year = pd.Timestamp.now().year
    for year in df["Year"].unique():
        assert year > 1945, f"Year {year} in is not greater than 1945."
        assert year <= current_year, f"Year {year} exceeds the current year."


def test_missing_years():
    levels = ["county", "msa"]
    for level in levels:
        tensed_features = list_tensed_features(level=level)

        if level == "msa":
            data = MSADataGrabber()
        else:
            data = DataGrabber()

        data.get_features_long(tensed_features)

        for feature in tensed_features:
            check_years(data.long[feature])


def test_DataGrabber_data_types():
    data = DataGrabber()

    general_data_format_testing(data, features)


def test_MSADataGrabber_data_types():
    data_msa = MSADataGrabber()

    general_data_format_testing(data_msa, features_msa)


def test_feature_listing_runtime():
    features = list_available_features()
    tensed_features = list_tensed_features()
    interventions = list_interventions()
    outcomes = list_outcomes()

    assert len(features) > 2
    assert len(tensed_features) > 2
    assert len(interventions) > 2
    assert len(outcomes) > 2


def test_no_ma_strings_in_features():
    features = list_available_features()
    assert all(not feature.endswith("_ma") for feature in features)


def test_ma_strings_in_features():
    feature_ma = list_available_features("msa")
    assert all(feature.endswith("_ma") for feature in feature_ma)


data_msa = MSADataGrabber()
data_msa.get_features_long(features_msa)


def test_GeoFIPS_ma_column_values():
    for feature in features_msa:
        data_msa.long[feature]["GeoFIPS"]
        column_values = data_msa.long[feature]["GeoFIPS"]

        assert all(value > 9999 and str(value)[-1] == "0" for value in column_values)
