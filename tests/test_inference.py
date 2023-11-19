import os
import random
import pytest 
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.data_grabber import list_interventions, list_outcomes, DataGrabber
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()

# TODO this needs to be parametrized by outcome datasets,
# intervention datasets and forward shifts
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

#     num_iterations = 1

#     interventions = list_interventions()
#     outcomes = list_outcomes()
#     shifts = [1,2,3]
    
#     intervention = [random.choice(interventions)]
#     outcome = [random.choice(outcomes)]
   


#     dg = DataGrabber()
#     dg.get_features_std_long(list_interventions())

#     intervention_variables = []

#     for intervention_dataset in intervention:
#         intervention_variable = dg.std_long[intervention_dataset].columns[-1]


#     for intervention_dataset, intervention_variable in zip(interventions, intervention_variables):
#         for outcome_dataset in outcome:
#             for forward_shift in shifts:
                
#                 guide_name = (f"{intervention_dataset}_{outcome_dataset}_{forward_shift}")
#                 file_path = os.path.join(root, "data/model_guides", f"{guide_name}_guide.pkl")

#                 if not os.path.exists(file_path):
                 
#                     model = InteractionsModel(outcome_dataset=outcome_dataset,
#                             intervention_dataset=intervention_dataset,
#                             intervention_variable=intervention_variable,
#                             forward_shift = forward_shift,
#                             num_iterations= num_iterations)
                    
#                     model.train_interactions_model()

                    




# def test_guide_presence():

#     interventions = list_interventions()
#     outcomes = list_outcomes()
#     shifts = [1,2,3]

#     N_combinations = len(interventions) * len(outcomes) * len(shifts)

#     root = find_repo_root()
#     _dir = os.path.join(root, "data", "model_guides")
#     files = [f for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f))]
#     num_files = len(files)

#     assert num_files == N_combinations + 2, f"{N_combinations + 2 - num_files} guides are missing"
#     #two extra files: .gitkeep and .training.log

                 
