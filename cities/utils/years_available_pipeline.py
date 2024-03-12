import os

import dill

from cities.modeling.modeling_utils import prep_wide_data_for_inference
from cities.utils.data_grabber import (find_repo_root, list_interventions,
                                       list_outcomes)

root = find_repo_root()
interventions = list_interventions()
outcomes = list_outcomes()


for intervention in interventions:
    for outcome in outcomes:
        # intervention = "spending_HHS"
        # outcome = "gdp"
        data = prep_wide_data_for_inference(
            outcome_dataset=outcome,
            intervention_dataset=intervention,
            forward_shift=3,  # shift doesn't matter here, as long as data exists
        )
        data_slim = {key: data[key] for key in ["years_available", "outcome_years"]}

        assert len(data_slim["years_available"]) > 2
        file_path = os.path.join(
            root, "data/years_available", f"{intervention}_{outcome}.pkl"
        )
        print(file_path)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                dill.dump(data_slim, f)
