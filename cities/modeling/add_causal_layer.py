import contextlib
import copy
from typing import Dict, List, Tuple

import torch


# TODO add softmax and treat linear pred as logit if output categorical



@contextlib.contextmanager
def AddCausalLayer(
    model, 
    dataset: Dict,
    causal_layer: Dict,   # keys required to be the downstream nodes
      # TODO type hint where mypy doesn't complain about forward
):
    
    
    causal_layer_variable_names = list(causal_layer.keys())+ list(causal_layer.values())
    assert all([name in dataset['categorical'].keys() or name in dataset['continuous'].keys() for name in causal_layer_variable_names])
    
    data_types = {key: 'categorical' if key in dataset['categorical'].keys() else 'continuous' for key in causal_layer_variable_names}

    print(data_types)

    old_forward = model.forward

    def new_forward(**kwargs):

        new_kwargs = copy.deepcopy(kwargs)
        

    #     new_kwargs, indexing_dictionaries = replace_categorical_with_combos(
    #         kwargs, interaction_tuples
    #     )

    #     model.indexing_dictionaries = indexing_dictionaries
    #     model.new_kwargs = new_kwargs
    #     old_forward(**new_kwargs)

    # model.forward = new_forward

    yield

    # model.forward = old_forward
