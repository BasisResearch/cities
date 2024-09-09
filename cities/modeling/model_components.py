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


def check_categorical_is_subset_of_levels(categorical, categorical_levels):

    assert set(categorical.keys()).issubset(set(categorical_levels.keys()))

    # # TODO should these be subsets or can we only check lengths?
    # if not all([set(torch.unique(v)).issubset(set(categorical_levels[k])) for k, v in categorical.items()]):
    #     raise ValueError("The passed categorical values are from a superset of the provided levels."
    #                      " See note about the categorical_levels param in the __init__ docstring.")

    return True


def get_categorical_levels(categorical):
    """
    Assumes that no levels are missing from the categorical data, and constructs the levels from the unique values.
    This should only be used with supersets of all data (so that every data subset will have its levels represented
    in the levels returned here.
    """
    return {name: torch.unique(categorical[name]) for name in categorical.keys()}


def categorical_contribution(
    categorical: Dict[str, torch.Tensor],
    child_name: str,
    leeway: float,
    categorical_levels: Dict[str, torch.Tensor],
) -> torch.Tensor:

    check_categorical_is_subset_of_levels(categorical, categorical_levels)

    categorical_names = list(categorical.keys())

    weights_categorical_outcome = {}
    objects_cat_weighted = {}

    for name in categorical_names:
        weights_categorical_outcome[name] = pyro.sample(
            f"weights_categorical_{name}_{child_name}",
            dist.Normal(0.0, leeway).expand(categorical_levels[name].shape).to_event(1),
        )

        weights_batch_dims = weights_categorical_outcome[name].shape[:-1]
        lwbd = len(weights_batch_dims)

        # FIXME
        #  Problem: if the categorical variable is being conditioned on per unit (e.g. in an ITE), then
        #   it won't be plated according to the sample plate, which means that the gather below has to
        #   broadcast those levels across the plated weights_categorical_outcome samples.
        # HACKy solution:
        # We can tell when this happens if weights_categorical_outcome batch shape doesn't match
        #  the categorical shape. In that case, we have to manually tile to broadcast the gather.
        # Otherwise, we use the view method.
        conditioned = categorical[name].ndim == 1  # we are conditioning.
        if conditioned:
            weight_indices = torch.tile(
                categorical[name].view(*((1,) * lwbd), -1),
                dims=(*weights_batch_dims, 1)
            )
        else:
            weight_indices = categorical[name].view(*weights_batch_dims, -1)

        objects_cat_weighted[name] = torch.gather(
            weights_categorical_outcome[name],
            dim=-1,
            index=weight_indices
        )

        # FIXME HACK any outer plates will tacked onto weights_categorical_outcome AFTER the event_dim, meaning
        # e.g. for outer plate 13, and data plate 816, and categorical levels 10, we'd have weights.shape == (13, 1, 10)
        # This propagates through the gather, but we want the final to be (13, 816) and not (13, 1, 816).
        if (not conditioned) and (objects_cat_weighted[name].shape[-lwbd:] != categorical[name].shape[-lwbd:]):
            # Note that this is always -2 b/c we're squeezing the single extra dimension resulting from the event dim
            #  above.
            assert objects_cat_weighted[name].shape[-2] == 1
            objects_cat_weighted[name] = objects_cat_weighted[name].squeeze(-2)

    values = list(objects_cat_weighted.values())

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
    categorical_levels: Dict[str, torch.Tensor],
    observations: Optional[torch.Tensor] = None,
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
    categorical_levels: Dict[str, torch.Tensor],
    observations: Optional[torch.Tensor] = None,
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
    categorical_levels: Dict[str, torch.Tensor],
    observations: Optional[torch.Tensor] = None,
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
