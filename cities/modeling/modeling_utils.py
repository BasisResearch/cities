import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

from cities.utils.data_grabber import (
    DataGrabber,
    list_available_features,
    list_tensed_features,
)


def train_interactions_model(
    model, num_iterations=2500, lr=0.01, print_interval=100, model_args=None
):
    pyro.clear_param_store()
    guide = AutoNormal(model)

    svi = SVI(model=model, guide=guide, optim=Adam({"lr": lr}), loss=Trace_ELBO())

    losses = []
    for step in range(num_iterations):
        loss = svi.step(*model_args)
        losses.append(loss)
        if step % print_interval == 0:
            print("[iteration %04d] loss: %.4f" % (step + 1, loss))

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

    # TODO revise once transformations are available
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
