from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim.optim import ClippedAdam

from cities.utils.data_grabber import (
    DataGrabber,
    list_available_features,
    list_tensed_features,
)


def prep_wide_data_for_inference(
    outcome_dataset: str, intervention_dataset: str, forward_shift: int
):
    """
    Prepares wide-format data for causal inference modeling.

    Parameters:
        - outcome_dataset (str): Name of the outcome variable.
        - intervention_dataset (str): Name of the intervention variable.
        - forward_shift (int): Number of time steps to shift the outcome variable for prediction.

    Returns:
        dict: A dictionary containing the necessary inputs for causal inference modeling.

    The function performs the following steps:
        1. Identifies available device (GPU if available, otherwise CPU), to be used with tensors.
        2. Uses a DataGrabber class to obtain standardized wide-format data.
        3. Separates covariate datasets into time series (tensed) and fixed covariates.
        4. Loads the required transformed features.
        5. Merges fixed covariates into a joint dataframe based on a common ID column.
        6. Ensures that the GeoFIPS (geographical identifier) is consistent across datasets.
        7. Shifts the outcome variable forward by the specified number of time steps determined by forward_shift.
        8. Extracts common years for which both intervention and outcome data are available.
        9. Prepares tensors for input features (x), interventions (t), and outcomes (y).
        10. Creates indices for states and units, preparing them as tensors.
        11. Validates the shapes of the tensors.
        12. Constructs a dictionary containing model arguments and prepared tensors.

    Example usage:
        prep_data = prep_wide_data_for_inference("outcome_data", "intervention_data", 2)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dg = DataGrabber()

    tensed_covariates_datasets = [
        var
        for var in list_tensed_features()
        if var not in [outcome_dataset, intervention_dataset]
    ]
    fixed_covariates_datasets = [
        var
        for var in list_available_features()
        if var
        not in tensed_covariates_datasets + [outcome_dataset, intervention_dataset]
    ]

    features_needed = [
        outcome_dataset,
        intervention_dataset,
    ] + fixed_covariates_datasets

    dg.get_features_std_wide(features_needed)

    intervention = dg.std_wide[intervention_dataset]
    outcome = dg.std_wide[outcome_dataset]

    # put covariates in one df as columns, dropping repeated ID columns
    f_covariates = {
        dataset: dg.std_wide[dataset] for dataset in fixed_covariates_datasets
    }
    f_covariates_joint = f_covariates[fixed_covariates_datasets[0]]
    for dataset in f_covariates.keys():
        if dataset != fixed_covariates_datasets[0]:
            if "GeoName" in f_covariates[dataset].columns:
                f_covariates[dataset] = f_covariates[dataset].drop(columns=["GeoName"])
            f_covariates_joint = f_covariates_joint.merge(
                f_covariates[dataset], on=["GeoFIPS"]
            )

    assert f_covariates_joint["GeoFIPS"].equals(intervention["GeoFIPS"])

    # extract data for which intervention and outcome overlap
    outcome.drop(columns=["GeoFIPS", "GeoName"], inplace=True)
    intervention.drop(columns=["GeoFIPS", "GeoName"], inplace=True)
    outcome_shifted = outcome.rename(lambda x: str(int(x) - forward_shift), axis=1)
    years_available = [
        year for year in intervention.columns if year in outcome_shifted.columns
    ]
    intervention = intervention[years_available]
    outcome_shifted = outcome_shifted[years_available]

    assert intervention.shape == outcome_shifted.shape

    unit_index = pd.factorize(f_covariates_joint["GeoFIPS"].values)[0]
    state_index = pd.factorize(f_covariates_joint["GeoFIPS"].values // 1000)[0]

    # prepare tensors
    x = torch.tensor(
        f_covariates_joint.iloc[:, 2:].values, dtype=torch.float32, device=device
    )
    x = x.unsqueeze(1).unsqueeze(1).permute(2, 3, 1, 0)

    t = torch.tensor(intervention.values, dtype=torch.float32, device=device)
    t = t.unsqueeze(1).unsqueeze(1).permute(3, 1, 2, 0)

    y = torch.tensor(outcome_shifted.values, dtype=torch.float32, device=device)
    y = y.unsqueeze(1).unsqueeze(1).permute(3, 1, 2, 0)

    state_index = torch.tensor(state_index, dtype=torch.int, device=device)
    unit_index = torch.tensor(unit_index, dtype=torch.int, device=device)

    N_t = y.shape[0]
    N_cov = x.shape[1]
    N_s = state_index.unique().shape[0]
    N_u = unit_index.unique().shape[0]

    assert x.shape == (1, N_cov, 1, N_u)
    assert y.shape == (N_t, 1, 1, N_u)
    assert t.shape == (N_t, 1, 1, N_u)

    model_args = (N_t, N_cov, N_s, N_u, state_index, unit_index)

    int_year_available = [int(year) for year in years_available]
    return {
        "model_args": model_args,
        "x": x,
        "t": t,
        "y": y,
        "years_available": int_year_available,
        "outcome_years": [str(year + forward_shift) for year in int_year_available],
    }


def train_interactions_model(
    conditioned_model: Callable,
    model_args,
    num_iterations: int = 1000,
    plot_loss: bool = True,
    print_interval: int = 100,
    lr: float = 0.01,
):
    guide = None
    pyro.clear_param_store()

    guide = AutoNormal(conditioned_model)

    svi = SVI(
        model=conditioned_model,
        guide=guide,
        optim=ClippedAdam({"lr": lr}),
        loss=Trace_ELBO(),
    )

    losses = []
    for step in range(num_iterations):
        loss = svi.step(*model_args)
        losses.append(loss)
        if step % print_interval == 0:
            print("[iteration %04d] loss: %.4f" % (step + 1, loss))

    if plot_loss:
        plt.plot(range(num_iterations), losses, label="Loss")
        plt.show()

    return guide


# reverting the standardization using the scaler is necessary, as doing this using the obvious formula
# leads to some numerical issues and inaccuracies
