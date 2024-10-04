from typing import Dict, List, Optional, Tuple

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

        if len(weights_categorical_outcome[name].shape) > 1:
            weights_categorical_outcome[name] = weights_categorical_outcome[
                name
            ].squeeze(-2)

        final_nonevent_shape = torch.broadcast_shapes(
            categorical[name].shape[:-1], weights_categorical_outcome[name].shape[:-1]
        )
        expanded_weight_indices = categorical[name].expand(*final_nonevent_shape, -1)
        expanded_weights = weights_categorical_outcome[name].expand(
            *final_nonevent_shape, -1
        )

        objects_cat_weighted[name] = torch.gather(
            expanded_weights, dim=-1, index=expanded_weight_indices
        )

        # weight_indices = categorical[name].expand(
        #     *weights_categorical_outcome[name].shape[:-1], -1
        # )

        # objects_cat_weighted[name] = torch.gather(
        #     weights_categorical_outcome[name], dim=-1, index=weight_indices
        # )

    values = list(objects_cat_weighted.values())

    categorical_contribution_outcome = torch.stack(values, dim=0).sum(dim=0)

    return categorical_contribution_outcome


def continuous_contribution(
    continuous: Dict[str, torch.Tensor],
    child_name: str,
    leeway: float,
) -> torch.Tensor:

    contributions = torch.zeros(1)

    bias_continuous = pyro.sample(
        f"bias_continuous_{child_name}",
        dist.Normal(0.0, leeway),
    )

    for key, value in continuous.items():

        weight_continuous = pyro.sample(
            f"weight_continuous_{key}_to_{child_name}",
            dist.Normal(0.0, leeway),
        )

        contribution = weight_continuous * value
        contributions = contribution + contributions

    contributions = bias_continuous + contributions

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
        child_continuous_parents, child_name, leeway=leeway
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
            continuous_contribution_to_child + categorical_contribution_to_child,
            event_dim=0,
        )

        child_observed = pyro.sample(  # type: ignore
            f"{child_name}",
            dist.Normal(mean_prediction_child, sigma_child),
            obs=observations,
        )

    return child_observed


def add_linear_component_continuous_interactions(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    continous_interaction_pairs: List[Tuple[str, str]],
    leeway: float,
    data_plate,
    categorical_levels: Dict[str, torch.Tensor],
    observations: Optional[torch.Tensor] = None,
) -> torch.Tensor:

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

        interaction_name = f"{interaction_pair[0]}_x_{interaction_pair[1]}"

        with data_plate:
            child_continuous_parents[interaction_name] = pyro.deterministic(
                interaction_name,
                child_continuous_parents[interaction_pair[0]]
                * child_continuous_parents[interaction_pair[1]],
                event_dim=0,
            )

    child_observed = add_linear_component(
        child_name=child_name,
        child_continuous_parents=child_continuous_parents,
        child_categorical_parents=child_categorical_parents,
        leeway=leeway,
        data_plate=data_plate,
        categorical_levels=categorical_levels,
        observations=observations,
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
            f"child_probs_{child_name}",
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
            f"child_probs_{child_name}",
            torch.sigmoid(mean_prediction_child),
            event_dim=0,
        )

        child_observed = pyro.sample(
            child_name, dist.Normal(child_probs, sigma_child), obs=observations
        )

    return child_observed


def add_ratio_component_continuous_interactions(
    child_name: str,
    child_continuous_parents: Dict[str, torch.Tensor],
    child_categorical_parents: Dict[str, torch.Tensor],
    continous_interaction_pairs: List[Tuple[str, str]],
    leeway: float,
    data_plate,
    categorical_levels: Dict[str, torch.Tensor],
    observations: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    for interaction_pair in continous_interaction_pairs:
        assert interaction_pair[0] in child_continuous_parents.keys()
        assert interaction_pair[1] in child_continuous_parents.keys()

        interaction_name = f"{interaction_pair[0]}_x_{interaction_pair[1]}"

        with data_plate:
            child_continuous_parents[interaction_name] = pyro.deterministic(
                interaction_name,
                child_continuous_parents[interaction_pair[0]]
                * child_continuous_parents[interaction_pair[1]],
                event_dim=0,
            )

    child_observed = add_ratio_component(
        child_name=child_name,
        child_continuous_parents=child_continuous_parents,
        child_categorical_parents=child_categorical_parents,
        leeway=leeway,
        data_plate=data_plate,
        categorical_levels=categorical_levels,
        observations=observations,
    )

    return child_observed
