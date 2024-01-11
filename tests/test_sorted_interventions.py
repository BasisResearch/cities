import os

import dill
import numpy as np

from cities.utils.data_grabber import DataGrabber, find_repo_root, list_interventions
from cities.utils.percentiles import transformed_intervention_from_percentile


def test_sorted_interventions_present():
    root = find_repo_root()

    interventions = list_interventions()

    file_path = os.path.join(
        root, "data/sorted_interventions", "interventions_sorted.pkl"
    )
    assert os.path.exists(file_path), f"The file {file_path} is missing."

    with open(file_path, "rb") as f:
        interventions_sorted = dill.load(f)

    for intervention in interventions:
        assert (
            intervention in interventions_sorted.keys()
        ), "Intervention missing, run `export_sorted_interventions()`."


def np_run(intervention, year, percentile):
    dg = DataGrabber()
    dg.get_features_std_wide([intervention])
    intervention_frame = dg.std_wide[intervention].copy().iloc[:, 2:]
    intervention_vector = intervention_frame[str(year)]
    value = np.percentile(intervention_vector, percentile)
    return value


def pandas_run(intervention, year, percentile):
    dg = DataGrabber()
    dg.get_features_std_wide([intervention])
    intervention_frame = dg.std_wide[intervention].copy().iloc[:, 2:]
    intervention_vector = intervention_frame[str(year)]
    value = intervention_vector.quantile(percentile / 100)
    return value


def test_transformed_intervention_from_percentile_accuracy():
    interventions = list_interventions()
    years = [2010, 2015, 2017]

    for intervention in interventions:
        for year in years:
            for percentile in [0, 25, 50, 75, 100]:
                assert np.allclose(
                    transformed_intervention_from_percentile(
                        intervention, year, percentile
                    ),
                    np_run(intervention, year, percentile),
                    rtol=0.01,
                )
                assert np.allclose(
                    transformed_intervention_from_percentile(
                        intervention, year, percentile
                    ),
                    pandas_run(intervention, year, percentile),
                    rtol=0.01,
                )
