import contextlib
from typing import Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch
from cities.modeling.simple_linear import SimpleLinear


import copy


def get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor]):
    N_categorical = len(categorical.keys())
    N_continuous = len(continuous.keys())

    if N_categorical > 0:
        n = len(next(iter(categorical.values())))
    elif N_continuous > 0:
        n = len(next(iter(continuous.values())))

    return N_categorical, N_continuous, n



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
    
    
    contributions = torch.zeros(1)

    for key, value in continuous.items():
        bias_continuous = pyro.sample(
            f"bias_continuous_{key}_{child_name}", dist.Normal(0.0, leeway),)

        weight_continuous = pyro.sample(
            f"weight_continuous_{key}_{child_name}_", dist.Normal(0.0, leeway),)

        contribution = bias_continuous + weight_continuous * value
        contributions = contribution + contributions
        
        
    return contributions
    


def add_linear_continuous_outcome(categorical_parents, 
                                                continuous_parents,
                                                child_name, 
                                                data_plate, n, leeway, 
                                                observations = None):

    categorical_contribution_to_child = torch.zeros(1, 1, 1, n)
    continuous_contribution_to_child = torch.zeros(1, 1, 1, n)

    if len(categorical_parents.keys()) > 0:

        categorical_contribution_to_child = categorical_contribution(
        categorical_parents, child_name, leeway
        )

    if len(continuous_parents.keys()) > 0:

        continuous_contribution_to_child = continuous_contribution(
            continuous_parents, child_name, leeway
        )

    sigma_child = pyro.sample(f"sigma_{child_name}", dist.Exponential(1.0))  # type: ignore


    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
            categorical_contribution_to_child
            + continuous_contribution_to_child,
            event_dim=0,
        )

        child_observed = pyro.sample(  # type: ignore
            f"{child_name}",
            dist.Normal(mean_prediction_child, sigma_child).to_event(1),
            obs=observations,
        )


def RegisterInput(model, input_names):
    def new_model(**kwargs):

        _kwargs = copy.deepcopy(kwargs)

        if "categorical" in input_names.keys():
            for key in input_names["categorical"].keys():
                _kwargs["categorical"][key] = pyro.sample(
                    key, dist.Delta(kwargs["categorical"][key])
                    )
        if "continuous" in input_names.keys():
            for key in kwargs["continuous"].keys():
                    _kwargs["continuous"][key] = pyro.sample(
                        key, dist.Delta(kwargs["continuous"][key])
                    )

        return model(**_kwargs)
    return new_model






class UnitsCausalModel(SimpleLinear):
    def __init__(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[
            torch.Tensor
        ] = None,  # init args kept for uniformity, consider deleting
        categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
        leeway=0.9,
    ):
        super().__init__(categorical, continuous, outcome, categorical_levels, leeway)

    def forward(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[torch.Tensor] = None,
        categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
        leeway=0.9,
    ):
        if categorical_levels is None:
            categorical_levels = self.categorical_levels

        _N_categorical, _N_continuous, n = get_n(categorical, continuous)
        data_plate = pyro.plate("data", size=n, dim=-1)

        #################
        # register input
        #################
        with data_plate:

            # continuous
            
            parcel_area = pyro.sample("parcel_area", dist.Normal(0, 1))#, obs = continuous['parcel_area'])

            limit_con = pyro.sample("limit_con", dist.Normal(0, 1), )#obs = continuous['limit_con'])


            # categorical
            # year = pyro.sample("year", dist.Categorical(torch.ones(len(categorical_levels['year']))),
            #                    obs = categorical['year'])
            
            # month = pyro.sample("month", dist.Categorical(torch.ones(len(categorical_levels['month']))),
            #                     obs = categorical['month'])
        



        # housing_units_cat_parents = {} #{'year': year, "month": month}
        # housing_units_con_parents = {'limit_con': limit_con, 'parcel_area': parcel_area}
        
        # add_linear_continuous_outcome(housing_units_cat_parents, 
        #                               housing_units_con_parents, 
        #                               'housing_units', data_plate, n, leeway,
        #                               observations=outcome)

