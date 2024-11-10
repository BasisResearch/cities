import copy

import torch

from cities.modeling.zoning_models.ts_model_components import (
    reshape_into_time_series,
    revert_to_original_shape,
)


def prepare_zoning_data_for_ts(data):

    unique_series = reshape_into_time_series(
        data["continuous"]["housing_units"],
        series_idx=data["categorical"]["census_tract"],
        time_idx=data["categorical"]["year"],
    )["unique_series"]

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

    data["init_state"] = data["reshaped"]["continuous"]["housing_units"][
        ..., 0
    ].unsqueeze(-1)
    data["init_idx"] = data["reshaped"]["categorical"]["census_tract"][
        ..., 0
    ].unsqueeze(-1)

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

    data["init_cumulative_state"] = data["reshaped"]["continuous"][
        "housing_units_cumulative"
    ][..., 0].unsqueeze(-1)

    subset_nonified = copy.deepcopy(data)
    subset_nonified["continuous"]["housing_units_cumulative_original"] = None
    subset_nonified["continuous"]["housing_units_cumulative"] = None
    subset_nonified["continuous"]["housing_units"] = None
    subset_nonified["continuous"]["housing_units_original"] = None

    subset_nonified["reshaped"]["continuous"]["housing_units"] = None
    subset_nonified["reshaped"]["continuous"]["housing_units_original"] = None
    subset_nonified["reshaped"]["continuous"][
        "housing_units_cumulative_original"
    ] = None
    subset_nonified["reshaped"]["continuous"]["housing_units_cumulative"] = None
