import os
import random
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.data_grabber import list_interventions, list_outcomes, DataGrabber
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()

# TODO this needs to be parametrized by outcome datasets,
# intervention datasets and forward shifts
def test_InteractionsModel():
    model = InteractionsModel(
        outcome_dataset="unemployment_rate",
        intervention_dataset="spending_commerce",
        intervention_variable="total_obligated_amount",
        forward_shift=2,
        num_iterations=10,
    )
    model.train_interactions_model()

    assert model.guide is not None
    assert model.model_args is not None
    assert model.model_conditioned is not None


def test_training_pipeline():
    num_iterations = 1

    interventions = list_interventions()
    outcomes = list_outcomes()
    shifts = [1,2,3]
    
    intervention = [random.choice(interventions)]
    outcomes = [random.choice(outcomes)]
    shifts = [random.choice(shifts)]


    dg = DataGrabber()
    dg.get_features_std_long(list_interventions())

    intervention_variables = []

    for intervention_dataset in interventions:
        intervention_variables.append(dg.std_long[intervention_dataset].columns[-1])


    for intervention_dataset, intervention_variable in zip(interventions, intervention_variables):
        for outcome_dataset in outcomes:
            for forward_shift in shifts:
                
                guide_name = (f"{intervention_dataset}_{outcome_dataset}_{forward_shift}")
                file_path = os.path.join(root, "data/model_guides", f"{guide_name}_guide.pkl")

                if not os.path.exists(file_path):
                 
                    model = InteractionsModel(outcome_dataset=outcome_dataset,
                            intervention_dataset=intervention_dataset,
                            intervention_variable=intervention_variable,
                            forward_shift = forward_shift,
                            num_iterations= num_iterations)
                    
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

    assert num_files == N_combinations + 2, f"{N_combinations + 2 - num_files} guides are missing"
    #two extra files: .gitkeep and .training.log

                 
