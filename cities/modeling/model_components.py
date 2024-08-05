from typing import Dict, Optional

import pyro
import pyro.distributions as dist
import torch


def get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor]):
    N_categorical = len(categorical)
    N_continuous = len(continuous)

    n_cat = next(iter(categorical.values())).shape[0] if N_categorical > 0 else None
    n_con = next(iter(continuous.values())).shape[0] if N_continuous > 0 else None

    if N_categorical > 0 and N_continuous > 0:
        if n_cat != n_con:
            raise ValueError("The number of categorical and continuous data points must be the same")


    n = n_cat if n_cat is not None else n_con

    if n is None:
        raise ValueError("Both categorical and continuous dictionaries are empty.")

    return N_categorical, N_continuous, n


def categorical_contribution(
    categorical: Dict[str, torch.Tensor],
    child_name: str,
    leeway: float,
    categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:

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


def continuous_contribution(
    continuous: Dict[str, torch.Tensor],
    child_name: str,
    leeway: float,
) -> torch.Tensor:

    contributions = torch.zeros(1)

    for key, value in continuous.items():
        bias_continuous = pyro.sample(
            f"bias_continuous_{key}_{child_name}",
            dist.Normal(0.0, leeway),
        )

        weight_continuous = pyro.sample(
            f"weight_continuous_{key}_{child_name}",
            dist.Normal(0.0, leeway),
        )

        contribution = bias_continuous + weight_continuous * value
        contributions = contribution + contributions

    return contributions


def add_linear_component(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    leeway: float,
    data_plate,
    observations: Optional[torch.Tensor] = None,
    categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:

    sigma_child = pyro.sample(
        f"sigma_{child_name}", dist.Exponential(1.0)
    )  # type: ignore

    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents,
        child_name,
        leeway,
        categorical_levels=categorical_levels,
    )

    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
            categorical_contribution_to_child + continuous_contribution_to_child,
            event_dim=0,
        )

        child_observed = pyro.sample(  # type: ignore
            f"{child_name}",
            dist.Normal(mean_prediction_child, sigma_child),
            obs=observations,
        )

    return child_observed


def add_logistic_component(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    leeway: float,
    data_plate,
    observations: Optional[torch.Tensor] = None,
    categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:

    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents,
        child_name,
        leeway,
        categorical_levels=categorical_levels,
    )

    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
            categorical_contribution_to_child + continuous_contribution_to_child,
            event_dim=0,
        )

        child_probs = pyro.deterministic(
            f"child_probs_{child_name}_{child_name}",
            torch.sigmoid(mean_prediction_child),
            event_dim=0,
        )

        child_observed = pyro.sample(
            f"{child_name}",
            dist.Bernoulli(child_probs),
            obs=observations,
        )

    return child_observed


def add_ratio_component(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    leeway: float,
    data_plate,
    observations: Optional[torch.Tensor] = None,
    categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:

    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway
    )

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents,
        child_name,
        leeway,
        categorical_levels=categorical_levels,
    )

    sigma_child = pyro.sample(f"sigma_{child_name}", dist.Exponential(40.0))

    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
            categorical_contribution_to_child + continuous_contribution_to_child,
            event_dim=0,
        )

        child_probs = pyro.deterministic(
            f"child_probs_{child_name}_{child_name}",
            torch.sigmoid(mean_prediction_child),
            event_dim=0,
        )

        child_observed = pyro.sample(
            child_name, dist.Normal(child_probs, sigma_child), obs=observations
        )

    return child_observed
