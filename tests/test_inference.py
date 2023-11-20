import os
import random
import pytest 
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.data_grabber import list_interventions, list_outcomes, DataGrabber
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()

# forward shifts caused some shape troubles,
# so we test for all of them at each random test
@pytest.mark.parametrize("shift", [1, 2, 3])
def test_InteractionsModel(shift):

    interventions = list_interventions()
    outcomes = list_outcomes()
    shifts = [1,2,3]
    
    intervention = [random.choice(interventions)]
    outcome = [random.choice(outcomes)]

    model = InteractionsModel(
        outcome_dataset="unemployment_rate",
        intervention_dataset="spending_commerce",
        forward_shift= shift,
        num_iterations=5,
        num_samples=5
    )
    model.train_interactions_model()

    model.sample_from_guide()

    assert model.model_args is not None ,f"Data prep failed for {intervention}, {outcome}."
    assert model.guide is not None, f"Training failed for {intervention}, {outcome}."
    assert model.model_conditioned is not None, f"Conditioning failed for {intervention}, {outcome}."


# def test_training_pipeline():

    interventions = list_interventions()
    outcomes = list_outcomes()
    shifts = [1,2,3]
    
    # running for all is too inefficient
    # let's just run for random
    intervention = [random.choice(interventions)]
    outcome = [random.choice(outcomes)]

    for intervention_dataset in intervention:
        for outcome_dataset in outcome:
            for forward_shift in shifts:
                
                guide_name = (f"{intervention_dataset}_{outcome_dataset}_{forward_shift}")
                file_path = os.path.join(root, "data/model_guides", f"{guide_name}_guide.pkl")
                 
                model = InteractionsModel(outcome_dataset=outcome_dataset,
                            intervention_dataset=intervention_dataset,
                            forward_shift = forward_shift,
                            num_iterations= 2)
                    
                model.train_interactions_model()
    


def test_guide_presence():

    interventions = list_interventions()
    outcomes = list_outcomes()
    shifts = [1,2,3]

    N_combinations = len(interventions) * len(outcomes) * len(shifts)

    root = find_repo_root()
    _dir = os.path.join(root, "data", "model_guides")
    files = [f for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f))]
    num_files = len(files)

    assert num_files ==  2 *  N_combinations + 2, f"{N_combinations + 2 - num_files} guides are missing"
    #two extra files: .gitkeep and .training.log

                 
