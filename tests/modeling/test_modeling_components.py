from typing import Dict

import pytest
import torch

from cities.modeling.model_components import get_n


@pytest.mark.parametrize(
    "categorical, continuous, expected",
    [
        # both categorical and continuous
        (
            {"cat1": torch.tensor([1, 2, 3, 4]), "cat2": torch.tensor([1, 2, 3, 4])},
            {"cont1": torch.tensor([0.5, 0.6, 0.7, 0.5])},
            (2, 1, 4),
        ),
        # only categorical
        ({"cat1": torch.tensor([1, 2, 3])}, {}, (1, 0, 3)),
        # only continuous
        ({}, {"cont1": torch.tensor([0.5, 0.6, 0.7, 0.8])}, (0, 1, 4)),
        # mixed size categorical
        (
            {
                "cat1": torch.tensor([1, 2, 3, 4, 5]),
                "cat2": torch.tensor([1, 2, 3, 4, 5]),
            },
            {},
            (2, 0, 5),
        ),
    ],
)
def test_get_n(
    categorical: Dict[str, torch.Tensor],
    continuous: Dict[str, torch.Tensor],
    expected: tuple,
):
    assert get_n(categorical, continuous) == expected


def test_get_n_error():
    with pytest.raises(
        ValueError,
        match="The number of categorical and continuous data points must be the same",
    ):
        get_n(
            {"cat1": torch.tensor([1, 2, 3, 4]), "cat2": torch.tensor([1, 2, 3, 4])},
            {"cont1": torch.tensor([0.5, 0.6, 0.5])},
        )
