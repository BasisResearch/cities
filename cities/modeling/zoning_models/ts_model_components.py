import copy
from typing import Dict, List, Optional, Tuple

import pyro
import pyro.distributions
import pyro.distributions as dist
import torch

from cities.modeling.model_components import (
    categorical_contribution,
    continuous_contribution,
)


def reshape_into_time_series(variable, series_idx, time_idx):

    if (
        variable.shape[-1] != series_idx.shape[0]
        or variable.shape[-1] != time_idx.shape[0]
    ):
        raise ValueError("The shapes of variable, series_idx, and time_idx must match.")

    unique_series = torch.unique(series_idx)
    unique_times = torch.unique(time_idx)

    num_series = unique_series.size(0)
    time_steps = unique_times.size(0)

    new_shape = (*variable.shape[:-1], num_series, time_steps)

    reshaped_variable = torch.empty(new_shape, dtype=variable.dtype)
    reshaped_variable[..., :] = -1042  # placeholder value for nan, to use with indices

    for i, series in enumerate(unique_series):
        for j, time in enumerate(unique_times):
            mask = (series_idx == series) & (time_idx == time)
            index = torch.where(mask)[0]
            if index.numel() > 0:
                reshaped_variable[..., i, j] = variable[..., index].squeeze(-1)

    for i, series_id in enumerate(unique_series):
        _, sorted_indices = torch.sort(time_idx[series_idx == series_id])
        sorted_outcomes = variable[..., series_idx == series_id][..., sorted_indices]
        assert torch.all(reshaped_variable[..., i, :] == sorted_outcomes)

        assert torch.all(reshaped_variable[..., i, :] != -1042)

    return {
        "reshaped_variable": reshaped_variable,
        "unique_series": unique_series,
        "unique_times": unique_times,
    }


def revert_to_original_shape(reshaped_output, series_idx, time_idx):

    unique_series = torch.unique(series_idx)
    unique_times = torch.unique(time_idx)

    original_shape = series_idx.shape[0]
    original_variable = torch.empty(original_shape, dtype=reshaped_output.dtype)
    original_variable[...] = -1042  # Placeholder, can be NaN if preferred

    # Map values from reshaped tensor back to the original shape
    for i, series in enumerate(unique_series):
        for j, time in enumerate(unique_times):
            # Find the index in the original shape where this (series, time) combination is located
            mask = (series_idx == series) & (time_idx == time)
            index = torch.where(mask)[0]
            if index.numel() > 0:
                original_variable[..., index] = reshaped_output[..., i, j]

    return original_variable


def prepare_zoning_data_for_ts(data):

    # TODO potentially redundant, check if needed donwstream
    # unique_series = reshape_into_time_series(
    #     data["continuous"]["housing_units"],
    #     series_idx=data["categorical"]["census_tract"],
    #     time_idx=data["categorical"]["year"],
    # )["unique_series"]

    # time-series reshaping of data: time at dim -1, series at dim -2
    data["reshaped"] = {}
    data["reshaped"]["continuous"] = {}
    data["reshaped"]["categorical"] = {}

    for key, val in data["continuous"].items():
        data["reshaped"]["continuous"][key] = reshape_into_time_series(
            val,
            series_idx=data["categorical"]["census_tract"],
            time_idx=data["categorical"]["year"],
        )["reshaped_variable"]

    for key, val in data["categorical"].items():
        data["reshaped"]["categorical"][key] = reshape_into_time_series(
            val,
            series_idx=data["categorical"]["census_tract"],
            time_idx=data["categorical"]["year"],
        )["reshaped_variable"]

    # needed for destandardization if hu are predicted
    data["housing_units_mean"] = data["reshaped"]["continuous"][
        "housing_units_original"
    ].mean()
    data["housing_units_std"] = data["reshaped"]["continuous"][
        "housing_units_original"
    ].std()

    # computing cumulative housing units and standardizing, in both shapes
    data["reshaped"]["continuous"]["housing_units_cumulative_original"] = torch.cumsum(
        data["reshaped"]["continuous"]["housing_units_original"], dim=-1
    )

    data["continuous"]["housing_units_cumulative_original"] = revert_to_original_shape(
        data["reshaped"]["continuous"]["housing_units_cumulative_original"],
        data["categorical"]["census_tract"],
        data["categorical"]["year"],
    )

    data["housing_units_cumulative_mean"] = data["continuous"][
        "housing_units_cumulative_original"
    ].mean()

    data["housing_units_cumulative_std"] = data["continuous"][
        "housing_units_cumulative_original"
    ].std()

    data["continuous"]["housing_units_cumulative"] = (
        data["continuous"]["housing_units_cumulative_original"]
        - data["housing_units_cumulative_mean"]
    ) / data["housing_units_cumulative_std"]

    data["reshaped"]["continuous"]["housing_units_cumulative"] = (
        reshape_into_time_series(
            data["continuous"]["housing_units_cumulative"],
            series_idx=data["categorical"]["census_tract"],
            time_idx=data["categorical"]["year"],
        )["reshaped_variable"]
    )

    # init states
    # index will be the same for all series
    data["init_idx"] = data["reshaped"]["categorical"]["census_tract"][
        ..., 0
    ].unsqueeze(-1)

    data["init_state"] = data["reshaped"]["continuous"]["housing_units"][
        ..., 0
    ].unsqueeze(-1)

    data["init_cumulative_state"] = data["reshaped"]["continuous"][
        "housing_units_cumulative"
    ][..., 0].unsqueeze(-1)

    # nonified version, to be used for prediction
    data_nonified = copy.deepcopy(data)
    data_nonified["continuous"]["housing_units_cumulative_original"] = None
    data_nonified["continuous"]["housing_units_cumulative"] = None
    data_nonified["continuous"]["housing_units"] = None
    data_nonified["continuous"]["housing_units_original"] = None

    data_nonified["reshaped"]["continuous"]["housing_units"] = None
    data_nonified["reshaped"]["continuous"]["housing_units_original"] = None
    data_nonified["reshaped"]["continuous"]["housing_units_cumulative_original"] = None
    data_nonified["reshaped"]["continuous"]["housing_units_cumulative"] = None

    data["n_series"] = data["reshaped"]["categorical"]["census_tract"].shape[-2]
    data["T"] = data["reshaped"]["categorical"]["year"].shape[-1]

    return data, data_nonified


def add_ar1_component_with_interactions(
    self,
    series_idx: int,
    time_idx: int,
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    continous_interaction_pairs: List[Tuple[str, str]],
    leeway: float,
    data_plate,
    categorical_levels: Dict[str, torch.Tensor],
    initial_observations=None,
    observations: Optional[torch.Tensor] = None,
    force_ts_reshape: bool = False,
) -> torch.Tensor:

    # -----------------------------------------
    # introducing interaction variables
    # -----------------------------------------

    if continous_interaction_pairs == [("all", "all")]:
        continous_interaction_pairs = [
            (key1, key2)
            for key1 in child_continuous_parents.keys()
            for key2 in child_continuous_parents.keys()
            if key1 != key2
        ]

    for interaction_pair in continous_interaction_pairs:
        assert interaction_pair[0] in child_continuous_parents.keys()
        assert interaction_pair[1] in child_continuous_parents.keys()

        interaction_name = (
            f"{interaction_pair[0]}_x_{interaction_pair[1]}_to_{child_name}"
        )

        with data_plate:
            child_continuous_parents[interaction_name] = pyro.deterministic(
                interaction_name,
                child_continuous_parents[interaction_pair[0]]
                * child_continuous_parents[interaction_pair[1]],
                event_dim=0,
            )

    # ---------------------------------------
    # reshaping data for time series modeling
    # with series at dim -2 and time at dim -1
    # with reshaping only if needed
    # ----------------------------------------

    unique_series = torch.unique(series_idx)
    unique_times = torch.unique(time_idx)
    no_series = unique_series.size(0)
    T = unique_times.size(0)

    series_plate = pyro.plate("series", no_series, dim=-2)
    time_plate = pyro.plate("time", T, dim=-1)

    shape_change_flag = False
    if self.continuous_parents_reshaped is not None:
        shape_change_flag = (
            len(child_continuous_parents["limit"].shape)
            > len(self.continuous_parents_reshaped["limit"].shape) - 1
        )

    if (
        self.continuous_parents_reshaped is not None
        and not force_ts_reshape
        and not shape_change_flag
    ):
        continuous_parents_reshaped = self.continuous_parents_reshaped
    else:
        continuous_parents_reshaped = {}
        for key in child_continuous_parents.keys():
            continuous_parents_reshaped[key] = reshape_into_time_series(
                child_continuous_parents[key], series_idx, time_idx
            )["reshaped_variable"]
        self.continuous_parents_reshaped = continuous_parents_reshaped

    if (
        self.categorical_parents_reshaped is not None
        and not force_ts_reshape
        and not shape_change_flag
    ):
        categorical_parents_reshaped = self.categorical_parents_reshaped
    else:
        categorical_parents_reshaped = {}
        for key in child_categorical_parents.keys():
            categorical_parents_reshaped[key] = reshape_into_time_series(
                child_categorical_parents[key], series_idx, time_idx
            )["reshaped_variable"]
        self.categorical_parents_reshaped = categorical_parents_reshaped

    if (
        self.outcome_reshaped is not None
        and not force_ts_reshape
        and not shape_change_flag
    ):
        outcome_reshaped = self.outcome_reshaped
    else:
        if observations is not None:
            outcome_reshaped = reshape_into_time_series(
                observations, series_idx, time_idx
            )["reshaped_variable"]
            self.outcome_reshaped = outcome_reshaped

    # ----------------------------
    # contributions of parents
    # ----------------------------

    sigma_child = pyro.sample(f"sigma_{child_name}", dist.Exponential(1.0))

    bias_arima_child = pyro.sample(f"bias_arima_{child_name}", dist.Normal(0.0, leeway))

    phi_arima_child = pyro.sample(f"phi_arima_{child_name}", dist.Uniform(0.0, 3.0))

    if child_continuous_parents != {}:
        continuous_contribution_to_child = continuous_contribution(
            continuous_parents_reshaped, child_name, leeway=leeway
        )
    else:
        continuous_contribution_to_child = torch.zeros(no_series, T)

    if child_categorical_parents != {}:
        categorical_contribution_to_child = categorical_contribution(
            categorical_parents_reshaped,
            child_name,
            leeway,
            categorical_levels=categorical_levels,
        )
    else:
        categorical_contribution_to_child = torch.zeros_like(
            continuous_contribution_to_child
        )

    # add contributions as deterministic sites

    with series_plate, time_plate:

        continuous_contribution_to_child = pyro.deterministic(
            f"continuous_contribution_to_{child_name}",
            continuous_contribution_to_child,
            event_dim=0,
        )

        categorical_contribution_to_child = pyro.deterministic(
            f"categorical_contribution_to_{child_name}",
            categorical_contribution_to_child,
            event_dim=0,
        )

    # AR scanning

    y_ts = {}
    y_exp = {}
    y_prev = {}

    with series_plate:
        y_prev[0] = torch.zeros(no_series, T)
        y_exp[0] = (
            bias_arima_child
            + continuous_contribution_to_child[..., 0].unsqueeze(-1)
            + categorical_contribution_to_child[..., 0].unsqueeze(-1)
        )
        y_ts[0] = pyro.sample(
            "y_0", dist.Normal(y_exp[0], sigma_child), obs=initial_observations
        )

    for t in range(1, T):
        with series_plate:
            y_prev[t] = y_ts[t - 1]
            y_exp[t] = (
                bias_arima_child
                + phi_arima_child * y_prev[t]
                + continuous_contribution_to_child[..., t].unsqueeze(-1)
                + categorical_contribution_to_child[..., t].unsqueeze(-1)
            )
            y_ts[t] = pyro.sample(
                f"y_{t}",
                dist.Normal(y_exp[t], sigma_child),
                obs=(
                    outcome_reshaped[..., t].unsqueeze(-1)
                    if observations is not None
                    else None
                ),
            )

    # putting together for convenience

    y_ts[0] = y_ts[0].expand(y_ts[1].shape)

    pyro.deterministic(f"expected_{child_name}", torch.cat(list(y_exp.values()), dim=1))

    predicted_child = pyro.deterministic(
        f"predicted_{child_name}", torch.cat(list(y_ts.values()), dim=1)
    )

    return predicted_child
