from typing import Dict
import pytest 
import torch

import cities.modeling.model_components as mc



# @pytest.mark.parametrize(
#     "categorical, continuous, expected",
#     [
#         # both categorical and continuous
#         (
#             {"cat1": torch.tensor([1, 2, 3]), "cat2": torch.tensor([1, 2, 3, 4])},
#             {"cont1": torch.tensor([0.5, 0.6, 0.7])},
#             (2, 1, 3)
#         ),
#         # only categorical
#         (
#             {"cat1": torch.tensor([1, 2, 3])},
#             {},
#             (1, 0, 3)
#         ),
#         # only continuous
#         (
#             {},
#             {"cont1": torch.tensor([0.5, 0.6, 0.7, 0.8])},
#             (0, 1, 4)
#         ),
#         # mixed size categorical
#         (
#             {"cat1": torch.tensor([1, 2, 3, 4, 5]), "cat2": torch.tensor([1, 2, 3])},
#             {},
#             (2, 0, 5)
#         ),
#     ]
# )
# def test_get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor], expected: tuple):
#     assert get_n(categorical, continuous) == expected
