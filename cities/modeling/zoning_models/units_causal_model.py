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


def add_linear_component(child_name: 'str', 
                    child_continuous_parents,
                    child_categorical_parents,
                    leeway,
                    data_plate,
                    observations = None):

    sigma_child = pyro.sample(f"sigma_{child_name}",
                    dist.Exponential(1.0))  # type: ignore

    continuous_contribution_to_child = continuous_contribution(
        child_continuous_parents, child_name, leeway)

    categorical_contribution_to_child = categorical_contribution(
        child_categorical_parents, child_name, leeway
    )

    with data_plate:

        mean_prediction_child = pyro.deterministic(  # type: ignore
            f"mean_outcome_prediction_{child_name}",
        categorical_contribution_to_child +
        continuous_contribution_to_child,
        event_dim=0,
        )

        child_observed = pyro.sample(  # type: ignore
        f"{child_name}",
        dist.Normal(mean_prediction_child, sigma_child),
        obs = observations
        )
    
    return child_observed


    

def categorical_interaction_variable(interaction_list: List[torch.Tensor]):

    assert len(interaction_list) > 1

    for tensor in interaction_list:
        assert tensor.shape == interaction_list[0].shape

        stacked_tensor = torch.stack(interaction_list, dim=-1)

        unique_pairs, inverse_indices = torch.unique(
            stacked_tensor, return_inverse=True, dim=0
        )

        unique_combined_tensor = inverse_indices.reshape(
            interaction_list[0].shape
        )

        indexing_dictionary = {
            tuple(pair.tolist()): i for i, pair in enumerate(unique_pairs)
        }

    return unique_combined_tensor, indexing_dictionary


class UnitsCausalModel(pyro.nn.PyroModule):
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
        super().__init__()

        self.leeway = leeway

        self.N_categorical, self.N_continuous, n = get_n(categorical, continuous)

        # you might need and pass further the original
        #  categorical levels of the training data
        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = dict()
            for name in categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])
        else:
            self.categorical_levels = categorical_levels



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
        # register 
        #################
        with data_plate:

            limit_con = pyro.sample("limit_con", dist.Normal(0, 1),
                                    obs = continuous['limit_con'])
            parcel_area = pyro.sample("parcel_area", dist.Normal(0, 1),
                                    obs = continuous['parcel_area'])

        
            year = pyro.sample("year", dist.Categorical(torch.ones(len(categorical_levels['year']))),
                                obs = categorical['year'])
            
            month = pyro.sample("month", dist.Categorical(torch.ones(len(categorical_levels['month']))),
                                 obs = categorical['month'])
        

        # _____________________________
        # regression for housing units
        # _____________________________

    
        housing_units_continuous_parents = {'limit_con': limit_con, 'parcel_area': parcel_area}
        housing_units_categorical_parents = {'year': year, "month": month}
        
        housing_units = add_linear_component(child_name =  "housing_units",
                            child_continuous_parents= housing_units_continuous_parents,
                            child_categorical_parents= housing_units_categorical_parents,
                            leeway=leeway,
                            data_plate=data_plate,
                            observations = outcome,)

























#     old_forward = model.forward

#     def new_forward(**kwargs):
#         new_kwargs = kwargs.copy()

#         new_kwargs, indexing_dictionaries = replace_categorical_with_combos(
#             kwargs, interaction_tuples
#         )

#         model.indexing_dictionaries = indexing_dictionaries
#         model.new_kwargs = new_kwargs
#         old_forward(**new_kwargs)

#     model.forward = new_forward

#     yield

#     model.forward = old_forward



           
    
            
            # for key, value in housing_units_con_parents.items():
            #     bias_continuous = pyro.sample(
            #         f"bias_continuous_{key}_{'housing_units'}", dist.Normal(0.0, leeway),)

            #     weight_continuous = pyro.sample(
            #         f"weight_continuous_{key}_{child_name}_", dist.Normal(0.0, leeway),)

            #     contribution = bias_continuous + weight_continuous * value
                #contributions = contribution + contributions
                
                
            # pyro.deterministic("contributions", contributions, event_dim=0)

            # housing_units_a = pyro.sample("housing_units_a", 
            #                               dist.Normal(limit_con.sum(), 1))

            # housing_units_cat_parents = {} #{'year': year, "month": month}
            # housing_units_con_parents = {'limit_con': limit_con, 'parcel_area': parcel_area}
            
            # continuous_contribution_housing_units = continuous_contribution(
            #     housing_units_con_parents, 'housing_units_b', leeway)



        # add_linear_continuous_outcome(housing_units_cat_parents, 
        #                                 housing_units_con_parents, 
        #                                 'housing_units', data_plate, n, leeway,
        #                                 observations=outcome)





























# def add_linear_continuous_outcome(categorical_parents, 
#                                                 continuous_parents,
#                                                 child_name, 
#                                                 data_plate, n, leeway, 
#                                                 observations = None):

#     categorical_contribution_to_child = torch.zeros(1, 1, 1, n)
#     continuous_contribution_to_child = torch.zeros(1, 1, 1, n)

#     if len(categorical_parents.keys()) > 0:

#         categorical_contribution_to_child = categorical_contribution(
#         categorical_parents, child_name, leeway
#         )

#     if len(continuous_parents.keys()) > 0:

#         continuous_contribution_to_child = continuous_contribution(
#             continuous_parents, child_name, leeway
#         )

#     sigma_child = pyro.sample(f"sigma_{child_name}", dist.Exponential(1.0))  # type: ignore


#     with data_plate:

#         mean_prediction_child = pyro.deterministic(  # type: ignore
#             f"mean_outcome_prediction_{child_name}",
#             categorical_contribution_to_child
#             + continuous_contribution_to_child,
#             event_dim=0,
#         )

#         child_observed = pyro.sample(  # type: ignore
#             f"{child_name}",
#             dist.Normal(mean_prediction_child, sigma_child).to_event(1),
#             obs=observations,
#         )


# def RegisterInput(model, input_names):
#     def new_model(**kwargs):

#         _kwargs = copy.deepcopy(kwargs)

#         if "categorical" in input_names.keys():
#             for key in input_names["categorical"].keys():
#                 _kwargs["categorical"][key] = pyro.sample(
#                     key, dist.Delta(kwargs["categorical"][key])
#                     )
#         if "continuous" in input_names.keys():
#             for key in kwargs["continuous"].keys():
#                     _kwargs["continuous"][key] = pyro.sample(
#                         key, dist.Delta(kwargs["continuous"][key])
#                     )

#         return model(**_kwargs)
#     return new_model



