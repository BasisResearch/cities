import contextlib
import copy
from typing import Dict, List, Tuple

import torch


def replace_categorical_with_combos(
    data: Dict, interaction_tuples: List[Tuple[str, ...]]
):

    unique_combined_tensors = {}
    inverse_indices_tensors = {}
    indexing_dictionaries = {}

    data_copy = copy.deepcopy(data)

    for interaction_tuple in interaction_tuples:

        assert len(interaction_tuple) > 1

        tensors_to_stack = [data_copy["categorical"][key] for key in interaction_tuple]

        for tensor in tensors_to_stack:
            assert tensor.shape == tensors_to_stack[0].shape

        stacked_tensor = torch.stack(tensors_to_stack, dim=-1)

        unique_pairs, inverse_indices = torch.unique(
            stacked_tensor, return_inverse=True, dim=0
        )

        inverse_indices_tensors[interaction_tuple] = inverse_indices

        unique_combined_tensor = inverse_indices.reshape(
            data_copy["categorical"][interaction_tuple[0]].shape
        )

        unique_combined_tensors[interaction_tuple] = unique_combined_tensor

        indexing_dictionaries[interaction_tuple] = {
            tuple(pair.tolist()): i for i, pair in enumerate(unique_pairs)
        }

        data_copy["categorical"][
            f"{'_'.join(interaction_tuple)}"
        ] = unique_combined_tensor

        for key in interaction_tuple:
            data_copy["categorical"].pop(key, None)

    return data_copy, indexing_dictionaries


@contextlib.contextmanager
def AddCategoricalInteractions(
    model,  # TODO type hint where mypy doesn't complain about forward
    interaction_tuples: List[Tuple[str, ...]],
):

    old_forward = model.forward

    def new_forward(**kwargs):
        new_kwargs = kwargs.copy()

        new_kwargs, indexing_dictionaries = replace_categorical_with_combos(
            kwargs, interaction_tuples
        )

        model.indexing_dictionaries = indexing_dictionaries
        model.new_kwargs = new_kwargs
        old_forward(**new_kwargs)

    model.forward = new_forward

    yield

    model.forward = old_forward
