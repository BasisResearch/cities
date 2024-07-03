import contextlib
import copy
from typing import Dict

import pyro
import pyro.distributions as dist
import torch

from cities.modeling.simple_linear import get_n

# TODO add categorical outcome


def categorical_contribution(categorical, child_name, leeway, categorical_levels=None):

    categorical_names = list(categorical.keys())

    if categorical_levels is None:
        categorical_levels = {
            name: torch.unique(categorical[name]) for name in categorical_names
        }

    weights_categorical_outcome = {}
    objects_cat_weighted = {}

    for name in categorical_names:
        weights_categorical_outcome[name] = pyro.sample(
            f"weights_categorical_{name}_{child_name}",
            dist.Normal(0.0, leeway).expand(categorical_levels[name].shape).to_event(1),
        )

        objects_cat_weighted[name] = weights_categorical_outcome[name][
            ..., categorical[name]
        ]

    values = list(objects_cat_weighted.values())
    for i in range(1, len(values)):
        values[i] = values[i].view(values[0].shape)

    categorical_contribution_outcome = torch.stack(values, dim=0).sum(dim=0)

    return categorical_contribution_outcome


def continuous_contribution(continuous, child_name, leeway):
    continuous_stacked = torch.stack(list(continuous.values()), dim=0)

    bias_continuous = pyro.sample(
        f"bias_continuous_{child_name}",
        dist.Normal(0.0, leeway).expand([continuous_stacked.shape[-2]]).to_event(1),
    )

    weight_continuous = pyro.sample(
        f"weight_continuous_{child_name}",
        dist.Normal(0.0, leeway).expand([continuous_stacked.shape[-2]]).to_event(1),
    )

    continuous_contribution = bias_continuous.sum() + torch.einsum(
        "...cd, ...c -> ...d", continuous_stacked, weight_continuous
    )

    return continuous_contribution


@contextlib.contextmanager
def AddCausalLayer(
    model,
    model_kwargs: Dict,
    dataset: Dict,
    causal_layer: Dict,  # keys required to be the downstream nodes
    # TODO type hint where mypy doesn't complain about forward
):

    new_layer_variable_names = [
        item for sublist in causal_layer.values() for item in sublist
    ]
    causal_layer_variable_names = list(
        set(list(causal_layer.keys()) + new_layer_variable_names)
    )

    assert all(
        name in dataset["categorical"].keys() or name in dataset["continuous"].keys()
        for name in causal_layer_variable_names
    )

    data_types = {
        key: "categorical"
        for key in causal_layer_variable_names
        if key in dataset["categorical"].keys()
    }
    data_types.update(
        {
            key: "continuous"
            for key in causal_layer_variable_names
            if key in dataset["continuous"].keys()
        }
    )
    data_types[model_kwargs["outcome"]] = (
        "categorical"
        if model_kwargs["outcome"] in dataset["categorical"].keys()
        else "continuous"
    )

    old_forward = model.forward

    def new_forward(**kwargs):

        new_kwargs = copy.deepcopy(kwargs)  # for extended variable operations
        new_minimal_kwargs = copy.deepcopy(
            kwargs
        )  # to swap sampled values and pass to the original model forward

        N_categorical, N_continuous, n = get_n(
            new_kwargs["categorical"], new_kwargs["continuous"]
        )

        # add missing variables to new_kwargs
        for variable in causal_layer_variable_names:
            if (
                variable not in new_kwargs[data_types[variable]].keys()
                and variable != model_kwargs["outcome"]
            ):
                new_kwargs[data_types[variable]][variable] = dataset[
                    data_types[variable]
                ][variable]

        # TODO make layer_counter play nice with nested handlers, this is a placeholder really
        layer_counter = 1
        data_plate = pyro.plate(f"data_{layer_counter}", size=n, dim=-1)
        # TODO: make sure all variable obs can be in the same dim
        layer_counter += 1

        for child in causal_layer.keys():

            categorical_contribution_to_child = torch.zeros(1, 1, 1, n)
            continuous_contribution_to_child = torch.zeros(1, 1, 1, n)

            categorical_parents = {
                key: value
                for key, value in new_kwargs["categorical"].items()
                if key in causal_layer[child]
            }

            continuous_parents = {
                key: value
                for key, value in new_kwargs["continuous"].items()
                if key in causal_layer[child]
            }

            if len(categorical_parents.keys()) > 0:

                categorical_contribution_to_child = categorical_contribution(
                    categorical_parents, child, model.leeway
                )

            if len(continuous_parents.keys()) > 0:

                continuous_contribution_to_child = continuous_contribution(
                    continuous_parents, child, model.leeway
                )

            sigma_child = pyro.sample(f"sigma_{child}", dist.Exponential(1.0))  # type: ignore

            # observations = (
            #     new_kwargs[data_types[child]][child]
            #     if child != model_kwargs["outcome"]
            #     else new_kwargs["outcome"]
            # )

            observations = (
                new_kwargs[data_types[child]][child]
                if child != model_kwargs["outcome"]
                and new_kwargs[data_types[child]][child] is not None
                else (
                    new_kwargs["outcome"] if new_kwargs["outcome"] is not None else None
                )
            )

            with data_plate:

                # TODO categorical outcome

                mean_prediction_child = pyro.deterministic(  # type: ignore
                    f"mean_outcome_prediction_{child}",
                    categorical_contribution_to_child
                    + continuous_contribution_to_child,
                    event_dim=0,
                )

                child_observed = pyro.sample(  # type: ignore
                    f"{child}",
                    dist.Normal(mean_prediction_child, sigma_child),
                    obs=observations,
                )

            if child != model_kwargs["outcome"]:
                new_minimal_kwargs[data_types[child]][child] = child_observed
            else:
                new_minimal_kwargs["outcome"] = child_observed

        # return old_forward(**new_minimal_kwargs)
        return old_forward(**kwargs)

    model.forward = new_forward

    yield

    model.forward = old_forward
