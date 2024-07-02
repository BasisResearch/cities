import contextlib
import copy
from typing import Dict, List, Tuple

import torch
import pyro.distributions as dist
import pyro


from cities.modeling.simple_linear import get_n

# TODO add categorical outcome



def categorical_contribution(categorical, child_name, leeway, categorical_levels=None):
    
    categorical_names = list(categorical.keys())
    
    if categorical_levels is None:
        categorical_levels = {name: torch.unique(categorical[name]) for name in categorical_names}

    weights_categorical_outcome = {}
    objects_cat_weighted = {}

    for name in categorical_names:
        weights_categorical_outcome[name] = pyro.sample(
            f"weights_categorical_{name}_{child_name}",
            dist.Normal(0.0, leeway)
            .expand(categorical_levels[name].shape)
            .to_event(1),
        )

        objects_cat_weighted[name] = weights_categorical_outcome[name][..., categorical[name]]

    values = list(objects_cat_weighted.values())
    for i in range(1, len(values)):
        values[i] = values[i].view(values[0].shape)

    categorical_contribution_outcome = torch.stack(values, dim=0).sum(dim=0)

    return categorical_contribution_outcome








@contextlib.contextmanager
def AddCausalLayer(
    model, 
    model_kwargs: Dict,
    dataset: Dict,
    causal_layer: Dict,   # keys required to be the downstream nodes
      # TODO type hint where mypy doesn't complain about forward
):
    
    
    new_layer_variable_names = [item for sublist in causal_layer.values() for item in sublist]
    causal_layer_variable_names = list(set(list(causal_layer.keys())+new_layer_variable_names))
    
    assert all(
    name in dataset['categorical'].keys() or name in dataset['continuous'].keys()
    for name in causal_layer_variable_names
    )

    data_types = {key: 'categorical' for key in causal_layer_variable_names if key in dataset['categorical'].keys()}
    data_types.update({key: 'continuous' for key in causal_layer_variable_names if key in dataset['continuous'].keys()})
    data_types[model_kwargs['outcome']] = 'categorical' if model_kwargs['outcome'] in dataset['categorical'].keys() else 'continuous'

    old_forward = model.forward

    def new_forward(**kwargs):
        
    
        new_kwargs = copy.deepcopy(kwargs)
        
        # add missing variables to new_kwargs
        for variable in causal_layer_variable_names:            
                if (variable not in new_kwargs[data_types[variable]].keys() and variable != model_kwargs['outcome']):
                    new_kwargs[data_types[variable]][variable] = dataset[data_types[variable]][variable]

        for child in causal_layer.keys():
             
            categorical_parents = {key: value for key, value in new_kwargs['categorical'].items() if key in causal_layer[child]}
            continuous_parents = {key: value for key, value in new_kwargs['continuous'].items() if key in causal_layer[child]}

            N_categorical, N_continuous, n = get_n(categorical_parents, continuous_parents)

            sigma_child = pyro.sample("sigma", dist.Exponential(1.0)) # type: ignore
            categorical_contribution_child = torch.zeros(1, 1, 1, n) 

            if N_categorical > 0:
                categorical_contribution_to_child = categorical_contribution(categorical_parents, 
                                                        "child", model.leeway)

            
            







    #     new_kwargs, indexing_dictionaries = replace_categorical_with_combos(
    #         kwargs, interaction_tuples
    #     )

    #     model.indexing_dictionaries = indexing_dictionaries
    #     model.new_kwargs = new_kwargs
    #     old_forward(**new_kwargs)

    model.forward = new_forward
    #       
            
        
    yield

    model.forward = old_forward
