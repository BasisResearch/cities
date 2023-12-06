from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.preprocessing import StandardScaler

from cities.utils.data_grabber import (
    DataGrabber,
    list_available_features,
    list_tensed_features,
)


def prep_wide_data_for_inference(
    outcome_dataset: str, intervention_dataset: str, forward_shift: int
):
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
    year_min = max(
        intervention.columns[2:].astype(int).min(),
        outcome.columns[2:].astype(int).min(),
    )

    year_max = min(
        intervention.columns[2:].astype(int).max(),
        outcome.columns[2:].astype(int).max(),
    )

    assert all(intervention["GeoFIPS"] == outcome["GeoFIPS"])

    outcome_years_to_keep = [
        year
        for year in outcome.columns[2:]
        if year_min <= int(year) <= year_max + forward_shift
    ]

    outcome_years_to_keep = [
        year for year in outcome_years_to_keep if year in intervention.columns[2:]
    ]

    outcome = outcome[outcome_years_to_keep]

    # shift outcome `forward_shift` steps ahead
    # for the prediction task
    outcome_shifted = outcome.copy()

    for i in range(len(outcome_years_to_keep) - forward_shift):
        outcome_shifted.iloc[:, i] = outcome_shifted.iloc[:, i + forward_shift]

    years_to_drop = [
        f"{year}" for year in range(year_max - forward_shift + 1, year_max + 1)
    ]
    outcome_shifted.drop(columns=years_to_drop, inplace=True)

    intervention.drop(columns=["GeoFIPS", "GeoName"], inplace=True)
    intervention = intervention[outcome_shifted.columns]

    assert intervention.shape == outcome_shifted.shape

    years_available = outcome_shifted.columns.astype(int).values

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

    return {
        "model_args": model_args,
        "x": x,
        "t": t,
        "y": y,
        "years_available": years_available,
        "outcome_years": outcome_years_to_keep,
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
        model=conditioned_model, guide=guide, optim=Adam({"lr": lr}), loss=Trace_ELBO()
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


def revert_standardize_and_scale_scaler(
    transformed_values: Union[np.ndarray, List, pd.Series, float],
    year: int,
    variable_name: str,
) -> List:
    if not isinstance(transformed_values, np.ndarray):
        transformed_values = np.array(transformed_values)

    def inverse_sigmoid(y, scale=1 / 3):
        return -np.log((2 / (y + 1)) - 1) / scale

    # needed to avoid lint issues
    dg: DataGrabber

    # normally this will be deployed in a context in which dg already exists
    # and we want to avoid wasting time by reloading the data
    try:
        original_column = dg.wide[variable_name][str(year)].values
    except NameError:
        dg = DataGrabber()
        dg.get_features_wide([variable_name])
        original_column = dg.wide[variable_name][str(year)].values.reshape(-1, 1)

    # dg = DataGrabber()
    # dg.get_features_wide([variable_name])

    # original_column = dg.wide[variable_name][str(year)].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(original_column)

    inverted_values_sigmoid = inverse_sigmoid(transformed_values)
    inverted_values = scaler.inverse_transform(
        inverted_values_sigmoid.reshape(-1, 1)
    ).flatten()

    return inverted_values


def revert_prediction_df(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    df_copy = df.copy()

    for i in range(len(df)):
        df_copy.iloc[i, 1:] = revert_standardize_and_scale_scaler(
            df.iloc[i, 1:].tolist(), df.iloc[i, 0], variable_name
        )

    return df_copy
