import os
import random

import numpy as np
import pandas as pd

from cities.utils.data_grabber import (  # TODO: Change to CTDataGrabber() in the future
    CTDataGrabberCSV,
    DataGrabber,
    MSADataGrabber,
    list_available_features,
    list_interventions,
    list_outcomes,
    list_tensed_features,
)

features = list_available_features()
features_msa = list_available_features("msa")
features_ct = list_available_features("census_tract")


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
            data.wide["gdp"]["GeoFIPS"].nunique()
            == data.wide[feature]["GeoFIPS"].nunique()
        )
        assert (
            data.long["gdp"]["GeoFIPS"].nunique()
            == data.long[feature]["GeoFIPS"].nunique()
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


def test_non_emptiness_CTDataGrabber():
    os.chdir(os.path.dirname(os.getcwd()))
    data_ct = CTDataGrabberCSV()  # TODO: Change to CTDataGrabber() in the future

    data_ct.get_features_wide(features_ct)
    data_ct.get_features_std_wide(features_ct)
    data_ct.get_features_long(features_ct)
    data_ct.get_features_std_long(features_ct)

    for feature in features_ct:
        assert data_ct.wide[feature].shape[0] > 100
        assert data_ct.std_wide[feature].shape[1] < 100
        assert data_ct.long[feature].shape[0] > 100
        assert data_ct.std_long[feature].shape[1] == 4


def general_data_format_testing(data, features, level="county_msa"):
    assert features is not None

    data.get_features_wide(features)
    data.get_features_std_wide(features)
    data.get_features_long(features)
    data.get_features_std_long(features)

    for feature in features:

        assert data.wide[feature].iloc[:, 0].dtype == np.int64, (
            f"Wrong data type for '{feature}' in 'data.wide' at {level} level: "
            f"Expected np.int64, got {data.wide[feature].iloc[:, 0].dtype}"
        )
        assert data.wide[feature].iloc[:, 1].dtype == object, (
            f"Wrong data type for '{feature}' in 'data.wide' at {level} level: "
            f"Expected object, got {data.wide[feature].iloc[:, 1].dtype}"
        )

        assert data.std_wide[feature].iloc[:, 0].dtype == np.int64, (
            f"Wrong data type for '{feature}' in 'data.std_wide' at {level} level: "
            f"Expected np.int64, got {data.std_wide[feature].iloc[:, 0].dtype}"
        )
        assert data.std_wide[feature].iloc[:, 1].dtype == object, (
            f"Wrong data type for '{feature}' in 'data.std_wide' at {level} level: "
            f"Expected object, got {data.std_wide[feature].iloc[:, 1].dtype}"
        )

        assert data.long[feature].iloc[:, 0].dtype == np.int64, (
            f"Wrong data type for '{feature}' in 'data.long' at {level} level: "
            f"Expected np.int64, got {data.long[feature].iloc[:, 0].dtype}"
        )
        assert data.long[feature].iloc[:, 1].dtype == object, (
            f"Wrong data type for '{feature}' in 'data.long' at {level} level: "
            f"Expected object, got {data.long[feature].iloc[:, 1].dtype}"
        )

        assert data.std_long[feature].iloc[:, 0].dtype == np.int64, (
            f"Wrong data type for '{feature}' in 'data.std_long' at {level} level: "
            f"Expected np.int64, got {data.std_long[feature].iloc[:, 0].dtype}"
        )
        assert data.std_long[feature].iloc[:, 1].dtype == object, (
            f"Wrong data type for '{feature}' in 'data.std_long' at {level} level: "
            f"Expected object, got {data.std_long[feature].iloc[:, 1].dtype}"
        )

    for feature in features:
        if level == "county_msa":
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

        elif level == "census_tract":
            pass  # TODO: check whether the county number is correct as indicated by the CT number

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


# def check_years(df):
#     current_year = pd.Timestamp.now().year
#     for year in df["Year"].unique():
#         assert year > 1945, f"Year {year} in is not greater than 1945."
#         assert year <= current_year, f"Year {year} exceeds the current year."


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


def test_CTDataGrabber_data_types():
    data_ct = CTDataGrabberCSV()  # TODO: Change to CTDataGrabber() in the future

    general_data_format_testing(data_ct, features_ct, level="census_tract")


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


data_ct = CTDataGrabberCSV()  # TODO: Change to CTDataGrabber() in the future
data_ct.get_features_wide(features_ct)


def test_GeoFIPS_ct_column_values():
    for feature in features_ct:
        data_ct.wide[feature]["GeoFIPS"]
        column_values = data_ct.wide[feature]["GeoFIPS"]

        assert all(value > 999999999 for value in column_values)


def test_ct_data_grabber_fips_consistency():
    time_periods = ["pre_2020", "post_2020"]
    variables = list_available_features(level="census_tract")

    data_by_period = {}

    for ct_time_period in time_periods:
        data = CTDataGrabberCSV(ct_time_period=ct_time_period)
        data.get_features_wide(variables)
        data_by_period[ct_time_period] = {var: data.wide[var] for var in variables}

    compare_variable = random.choice(variables)

    for ct_time_period in time_periods:
        var_compare = data_by_period[ct_time_period][compare_variable]
        fips_compare = var_compare["GeoFIPS"].nunique()

        for variable in variables:
            var = data_by_period[ct_time_period][variable]
            fips_standard = var["GeoFIPS"].nunique()

            assert fips_compare == fips_standard, (
                f"The CT variables differ in the number of FIPS codes: {compare_variable} and {variable},"
                f"with {fips_compare} and {fips_standard} respectively!"
            )


current_year = pd.Timestamp.now().year


def check_years(df):
    for year in df["Year"].unique():
        assert year > 1945, f"Year {year} is not greater than 1945."
        assert year <= current_year, f"Year {year} exceeds the current year."


tensed_features_ct = list_tensed_features(level="census_tract")


def test_missing_years_census_tract():
    time_periods = ["pre_2020", "post_2020"]

    for time_period in time_periods:
        data = CTDataGrabberCSV(ct_time_period=time_period)
        data.get_features_long(tensed_features_ct)

        for feature in tensed_features_ct:
            check_years(data.long[feature])
