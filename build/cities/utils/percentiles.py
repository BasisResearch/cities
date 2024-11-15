import os

import dill as dill
import numpy as np

from cities.utils.data_grabber import DataGrabber, find_repo_root, list_interventions


def export_sorted_interventions():
    root = find_repo_root()

    interventions = list_interventions()
    dg = DataGrabber()

    dg.get_features_std_wide(interventions)

    interventions_sorted = {}
    for intervention in interventions:
        intervention_frame = dg.std_wide[intervention].copy().iloc[:, 2:]
        intervention_frame = intervention_frame.apply(
            lambda col: col.sort_values().values
        )
        assert (
            all(np.diff(intervention_frame[col]) >= 0)
            for col in intervention_frame.columns
        ), "A column is not increasing."
        interventions_sorted[intervention] = intervention_frame

    with open(
        os.path.join(root, "data/sorted_interventions", "interventions_sorted.pkl"),
        "wb",
    ) as f:
        dill.dump(interventions_sorted, f)


def transformed_intervention_from_percentile(intervention, year, percentile):
    root = find_repo_root()

    with open(
        os.path.join(root, "data/sorted_interventions", "interventions_sorted.pkl"),
        "rb",
    ) as f:
        interventions_sorted = dill.load(f)
    intervention_frame = interventions_sorted[intervention]

    if str(year) not in intervention_frame.columns:
        raise ValueError("Year not in intervention frame.")

    sorted_var = intervention_frame[str(year)]
    n = len(sorted_var)
    index = percentile * (n - 1) / 100

    lower_index = int(index)
    upper_index = lower_index + 1

    if lower_index == n - 1:
        return sorted_var[lower_index]

    interpolation_factor = index - lower_index
    interpolated_value = (1 - interpolation_factor) * sorted_var[
        lower_index
    ] + interpolation_factor * sorted_var[upper_index]

    return interpolated_value
