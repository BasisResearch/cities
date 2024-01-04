import os
import numpy as np

from cities.utils.data_grabber import (
    DataGrabber, MSADataGrabber,
    list_available_features,
    list_interventions,
    list_outcomes,
    list_tensed_features,
)

features = list_available_features()



def test_DataGrabber():
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
            data.wide["gdp"]["GeoFIPS"].nunique()
            == data.wide[feature]["GeoFIPS"].nunique()
        )
        assert (
            data.long["gdp"]["GeoFIPS"].nunique()
            == data.long[feature]["GeoFIPS"].nunique()
        )

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
    feature_ma = list_available_features('msa')
    assert all(feature.endswith("_ma") for feature in feature_ma)


feature_ma = list_available_features('msa')
data = MSADataGrabber()
data.get_features_long(feature_ma)

def test_GeoFIPS_column_values():
    for feature in feature_ma:

        data.long[feature]['GeoFIPS']
        column_values = data.long[feature]['GeoFIPS']
        
        assert all(value > 9999 and str(value)[-1] == '0' for value in column_values)