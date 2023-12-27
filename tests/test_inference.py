import os
import random

import pytest
import torch

from cities.modeling.model_interactions import InteractionsModel
from cities.queries.causal_insight import CausalInsight
from cities.utils.cleaning_utils import find_repo_root
from cities.utils.data_grabber import list_interventions, list_outcomes

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

root = find_repo_root()

interventions = list_interventions()
outcomes = list_outcomes()
shifts = [1, 2, 3]

# TODO add list_fips() function
# and parametrize with random fips
# and uniformly random intervention value
fips = 1003
intervened_value = 0.9

# running for all is too inefficient
# let's just run for random on a regular basis
# comment out for a full battery of tests
# intervention = [random.choice(interventions)]
# outcome = [random.choice(outcomes)]

interventions = [random.choice(interventions)]
outcomes = [random.choice(outcomes)]


# forward shifts caused some shape troubles,
# so we test for all of them at each random test
@pytest.mark.parametrize("intervention", interventions)
@pytest.mark.parametrize("outcome", outcomes)
@pytest.mark.parametrize("shift", shifts)
def test_smoke_InteractionsModel(intervention, outcome, shift):
    model = InteractionsModel(
        outcome_dataset="unemployment_rate",
        intervention_dataset="spending_commerce",
        forward_shift=shift,
        num_iterations=5,
        num_samples=5,
    )
    model.train_interactions_model()

    model.sample_from_guide()

    assert (
        model.model_args is not None
    ), f"Data prep failed for {intervention}, {outcome}."
    assert model.guide is not None, f"Training failed for {intervention}, {outcome}."
    assert (
        model.model_conditioned is not None
    ), f"Conditioning failed for {intervention}, {outcome}."


@pytest.mark.parametrize("intervention", interventions)
@pytest.mark.parametrize("outcome", outcomes)
@pytest.mark.parametrize("shift", shifts)
def test_smoke_training_pipeline(intervention, outcome, shift):
    # guide_name = f"{intervention}_{outcome}_{shift}"
    # file_path = os.path.join(root, "data/model_guides", f"{guide_name}_guide.pkl")

    model = InteractionsModel(
        outcome_dataset=outcome,
        intervention_dataset=intervention,
        forward_shift=shift,
        num_iterations=2,
    )

    model.train_interactions_model()


@pytest.mark.parametrize("intervention", interventions)
@pytest.mark.parametrize("outcome", outcomes)
@pytest.mark.parametrize("shift", shifts)
def test_smoke_guide_presence(intervention, outcome, shift):
    interventions = list_interventions()
    outcomes = list_outcomes()
    shifts = [1, 2, 3]

    N_combinations = len(interventions) * len(outcomes) * len(shifts)

    root = find_repo_root()
    _dir = os.path.join(root, "data", "model_guides")
    files = [f for f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, f))]
    num_files = len(files)

    assert (
        num_files == 2 * N_combinations + 2
    ), f"{N_combinations + 2 - num_files} guides are missing"
    # two extra files: .gitkeep and .training.log


@pytest.mark.parametrize("intervention", interventions)
@pytest.mark.parametrize("outcome", outcomes)
@pytest.mark.parametrize("shift", shifts)
def test_smoke_CausalInsigth(intervention, outcome, shift):
    ci = CausalInsight(
        outcome_dataset=outcome,
        intervention_dataset=intervention,
        num_samples=5,
    )

    ci.load_guide(forward_shift=2)

    assert ci.guide is not None, f"Guide not found for {intervention}, {outcome}."

    ci.generate_samples()

    assert ci.samples is not None, f"Sampling failed for {intervention}, {outcome}."

    ci.generate_tensed_samples()

    assert (
        ci.tensed_samples is not None
    ), f"Tensing failed for {intervention}, {outcome}."

    ci.get_fips_predictions(fips, intervened_value)

    assert (
        ci.predictions is not None
    ), f"Prediction failed for {intervention}, {outcome}."

    ci.plot_predictions(show_figure=False)

    assert (
        ci.predictions_plot is not None
    ), f"Plotting failed for {intervention}, {outcome}."
