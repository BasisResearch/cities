from typing import Dict, Optional

import pyro
import pyro.distributions as dist
import torch


def get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor]):
    N_categorical = len(categorical)
    N_continuous = len(continuous)

    # a but convoluted, but groups might be missing and sometimes
    # vars are allowed to be None
    n_cat = None
    if N_categorical > 0:
        for value in categorical.values():
            if value is not None:
                n_cat = value.shape[0]
                break

    n_con = None
    if N_continuous > 0:
        for value in continuous.values():
            if value is not None:
                n_con = value.shape[0]
                break

    if N_categorical > 0 and N_continuous > 0:
        if n_cat != n_con:
            raise ValueError(
                "The number of categorical and continuous data points must be the same"
            )

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

        weights_batch_dims = weights_categorical_outcome[name].shape[:-1]
        # n = categorical[name].shape[-1]

        # FIXME
        #  Problem: if the categorical variable is being conditioned on per unit (e.g. in an ITE), then
        #   it won't be plated according to the sample plate, which means that the gather below has to
        #   broadcast those levels across the plated weights_categorical_outcome samples.
        # HACKy solution:
        # We can tell when this happens if weights_categorical_outcome batch shape doesn't match
        #  the categorical shape. In that case, we have to manually tile to broadcast the gather.
        # Otherwise, we use the view method.
        if categorical[name].ndim == 1:  # we are conditioning.
            weight_indices = torch.tile(
                categorical[name].view(*((1,) * len(weights_batch_dims)), -1),
                dims=(*weights_batch_dims, 1)
            )
        else:
            weight_indices = categorical[name].view(*weights_batch_dims, -1)

        objects_cat_weighted[name] = torch.gather(
            weights_categorical_outcome[name],
            dim=-1,
            index=weight_indices
        )

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
