from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam  # type: ignore
from scipy.stats import spearmanr

import pyro
from cities.utils.data_grabber import (
    DataGrabber,
    list_available_features,
    list_tensed_features,
)


def drop_high_correlation(df, threshold=0.85):
    df_var = df.iloc[:, 2:].copy()
    correlation_matrix, _ = spearmanr(df_var)

    high_correlation_pairs = [
        (df_var.columns[i], df_var.columns[j])
        for i in range(df_var.shape[1])
        for j in range(i + 1, df_var.shape[1])
        if abs(correlation_matrix[i, j]) > threshold
        and abs(correlation_matrix[i, j]) < 1.0
    ]
    high_correlation_pairs = [
        (var1, var2) for var1, var2 in high_correlation_pairs if var1 != var2
    ]

    removed = set()
    print(
        f"Highly correlated pairs: {high_correlation_pairs}, second elements will be dropped"
    )
    for var1, var2 in high_correlation_pairs:
        assert var2 in df_var.columns
    for var1, var2 in high_correlation_pairs:
        if var2 in df_var.columns:
            removed.add(var2)
            df_var.drop(var2, axis=1, inplace=True)

    result = pd.concat([df.iloc[:, :2], df_var], axis=1)
    print(f"Removed {removed} due to correlation > {threshold}")
    return result


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

    f_covariates_joint = drop_high_correlation(f_covariates_joint)

    assert f_covariates_joint["GeoFIPS"].equals(intervention["GeoFIPS"])

    # This is for the downstream variable
    outcome_years_to_keep = [
        year
        for year in outcome.columns[2:]
        if str(int(year) - forward_shift) in intervention.columns[2:]
    ]

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
        "outcome_years": outcome_years_to_keep,
        "covariates_df": f_covariates_joint,
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
    pyro.clear_param_store()  # type: ignore

    guide = AutoNormal(conditioned_model)

    svi = SVI(
        model=conditioned_model,
        guide=guide,
        optim=Adam({"lr": lr}),  # type: ignore
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


def prep_data_for_interaction_inference(
    outcome_dataset, intervention_dataset, intervention_variable, forward_shift
):
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

    dg.get_features_std_long(list_available_features())
    dg.get_features_std_wide(list_available_features())

    year_min = max(
        dg.std_long[intervention_dataset]["Year"].min(),
        dg.std_long[outcome_dataset]["Year"].min(),
    )
    year_max = min(
        dg.std_long[intervention_dataset]["Year"].max(),
        dg.std_long[outcome_dataset]["Year"].max(),
    )
    outcome_df = dg.std_long[outcome_dataset].sort_values(by=["GeoFIPS", "Year"])

    # now we adding forward shift to the outcome
    # cleaning up and puting intervention/outcome in one df
    # and fixed covariates in another

    outcome_df[f"{outcome_dataset}_shifted_by_{forward_shift}"] = None

    geo_subsets = []
    for geo_fips in outcome_df["GeoFIPS"].unique():
        geo_subset = outcome_df[outcome_df["GeoFIPS"] == geo_fips].copy()
        # Shift the 'Value' column `forward_shift` in a new column
        geo_subset[f"{outcome_dataset}_shifted_by_{forward_shift}"] = geo_subset[
            "Value"
        ].shift(-forward_shift)
        geo_subsets.append(geo_subset)

    outcome_df = pd.concat(geo_subsets).reset_index(drop=True)

    outcome = outcome_df[
        (outcome_df["Year"] >= year_min)
        & (outcome_df["Year"] <= year_max + forward_shift)
    ]

    intervention = dg.std_long[intervention_dataset][
        (dg.std_long[intervention_dataset]["Year"] >= year_min)
        & (dg.std_long[intervention_dataset]["Year"] <= year_max)
    ]
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

    i_o_data = pd.merge(outcome, intervention, on=["GeoFIPS", "Year"])

    if "GeoName_x" in i_o_data.columns:
        i_o_data.rename(columns={"GeoName_x": "GeoName"}, inplace=True)
        columns_to_drop = i_o_data.filter(regex=r"^GeoName_[a-zA-Z]$")
        i_o_data.drop(columns=columns_to_drop.columns, inplace=True)

    i_o_data.rename(columns={"Value": outcome_dataset}, inplace=True)

    i_o_data["state"] = [code // 1000 for code in i_o_data["GeoFIPS"]]

    N_s = len(i_o_data["state"].unique())  # number of states
    i_o_data.dropna(inplace=True)

    i_o_data["unit_index"] = pd.factorize(i_o_data["GeoFIPS"].values)[0]
    i_o_data["state_index"] = pd.factorize(i_o_data["state"].values)[0]
    i_o_data["time_index"] = pd.factorize(i_o_data["Year"].values)[0]

    assert i_o_data["GeoFIPS"].isin(f_covariates_joint["GeoFIPS"]).all()

    f_covariates_joint.drop(columns=["GeoName"], inplace=True)
    data = i_o_data.merge(f_covariates_joint, on="GeoFIPS", how="left")

    assert not data.isna().any().any()

    time_index_idx = data.columns.get_loc("time_index")
    covariates_df = data.iloc[:, time_index_idx + 1 :].copy()
    covariates_df_sparse = covariates_df.copy()
    covariates_df_sparse["unit_index"] = data["unit_index"]
    covariates_df_sparse["state_index"] = data["state_index"]
    covariates_df_sparse.drop_duplicates(inplace=True)
    assert set(covariates_df_sparse["unit_index"]) == set(data["unit_index"])

    # get tensors

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    y = data[f"{outcome_dataset}_shifted_by_{forward_shift}"]
    y = torch.tensor(y, dtype=torch.float32, device=device)

    unit_index = torch.tensor(data["unit_index"], dtype=torch.int, device=device)
    unit_index_sparse = torch.tensor(
        covariates_df_sparse["unit_index"], dtype=torch.int
    )

    state_index = torch.tensor(data["state_index"], dtype=torch.int, device=device)
    state_index_sparse = torch.tensor(
        covariates_df_sparse["state_index"], dtype=torch.int
    )

    time_index = torch.tensor(data["time_index"], dtype=torch.int, device=device)
    intervention = torch.tensor(
        data[intervention_variable], dtype=torch.float32, device=device
    )

    covariates = torch.tensor(covariates_df.values, dtype=torch.float32, device=device)

    covariates_df_sparse.drop(columns=["unit_index", "state_index"], inplace=True)
    covariates_sparse = torch.tensor(
        covariates_df_sparse.values, dtype=torch.float32, device=device
    )

    N_cov = covariates.shape[1]  # number of covariates
    N_u = covariates_sparse.shape[0]  # number of units (counties)
    N_obs = len(y)  # number of observations
    N_t = len(time_index.unique())  # number of time points
    N_s = len(state_index.unique())  # number of states

    assert len(intervention) == len(y)
    assert len(unit_index) == len(y)
    assert len(state_index) == len(unit_index)
    assert len(time_index) == len(unit_index)
    assert covariates.shape[1] == covariates_sparse.shape[1]
    assert len(unit_index_sparse) == N_u

    return {
        "N_t": N_t,
        "N_cov": N_cov,
        "N_s": N_s,
        "N_u": N_u,
        "N_obs": N_obs,
        "unit_index": unit_index,
        "state_index": state_index,
        "time_index": time_index,
        "unit_index_sparse": unit_index_sparse,
        "state_index_sparse": state_index_sparse,
        "covariates": covariates,
        "covariates_sparse": covariates_sparse,
        "intervention": intervention,
        "y": y,
    }
