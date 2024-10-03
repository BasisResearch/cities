from typing import Dict

import pyro
import pytest
import torch

from cities.modeling.model_components import (
    add_linear_component,
    add_logistic_component,
    add_ratio_component,
    categorical_contribution,
    continuous_contribution,
    get_categorical_levels,
    get_n,
)


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


# setup for component tests
mock_data_cat = {"cat1": torch.tensor([2, 1, 0]), "cat2": torch.tensor([1, 0, 1])}
mock_data_cont = {
    "cont1": torch.tensor([1.0, 2.0, 3.0]),
    "cont2": torch.tensor([4.0, 5.0, 6.0]),
}
categorical_levels = {"cat1": torch.tensor([0, 1, 2]), "cat2": torch.tensor([0, 1])}


def test_categorical_contribution():

    with pyro.poutine.trace() as tr:
        cat_contribution = categorical_contribution(
            mock_data_cat,
            "child1",
            0.3,
            get_categorical_levels(mock_data_cat),
        )

        weights_1 = tr.trace.nodes["weights_categorical_cat1_child1"]["value"]
        assert weights_1.shape == (3,)

        weights_2 = tr.trace.nodes["weights_categorical_cat2_child1"]["value"]
        assert weights_2.shape == (2,)

        assert torch.equal(
            weights_1[mock_data_cat["cat1"]] + weights_2[mock_data_cat["cat2"]],
            cat_contribution,
        )


def test_continuous_contribution():

    with pyro.poutine.trace() as tr:
        cont_contribution = continuous_contribution(mock_data_cont, "child1", 0.5)

        bias = tr.trace.nodes["bias_continuous_child1"]["value"]
        weight_cont1 = tr.trace.nodes["weight_continuous_cont1_to_child1"]["value"]
        weight_cont2 = tr.trace.nodes["weight_continuous_cont2_to_child1"]["value"]

        assert bias.shape == torch.Size([])
        assert weight_cont1.shape == torch.Size([])
        assert weight_cont2.shape == torch.Size([])

        expected_contribution = (
            bias
            + weight_cont1 * mock_data_cont["cont1"]
            + weight_cont2 * mock_data_cont["cont2"]
        )

        assert torch.allclose(cont_contribution, expected_contribution)


def test_add_linear_component():

    data_plate = pyro.plate("data_plate", 3)

    with pyro.poutine.trace() as tr:
        add_linear_component(
            child_name="child1",
            child_continuous_parents=mock_data_cont,
            child_categorical_parents=mock_data_cat,
            leeway=0.5,
            data_plate=data_plate,
            observations=None,
            categorical_levels=categorical_levels,
        )

    sigma_child = tr.trace.nodes["sigma_child1"]["value"]
    mean_prediction_child = tr.trace.nodes["mean_outcome_prediction_child1"]["value"]

    sigma_child = tr.trace.nodes["sigma_child1"]["value"]
    mean_prediction_child = tr.trace.nodes["mean_outcome_prediction_child1"]["value"]

    assert sigma_child.shape == torch.Size([])
    assert mean_prediction_child.shape == torch.Size([3])

    weights_categorical = {}
    for name in mock_data_cat.keys():
        weights_categorical[name] = tr.trace.nodes[
            f"weights_categorical_{name}_child1"
        ]["value"]

    categorical_contrib = torch.zeros(3)
    for name, tensor in mock_data_cat.items():
        categorical_contrib += weights_categorical[name][..., tensor]

    continuous_contrib = torch.zeros(3)
    bias = tr.trace.nodes["bias_continuous_child1"]["value"]

    for key, value in mock_data_cont.items():
        weight = tr.trace.nodes[f"weight_continuous_{key}_to_child1"]["value"]
        continuous_contrib += weight * value
    continuous_contrib += bias

    expected_mean_prediction = categorical_contrib + continuous_contrib

    assert torch.allclose(mean_prediction_child, expected_mean_prediction, atol=1e-6)


def test_add_logistic_component():

    data_plate = pyro.plate("data_plate", 3)

    with pyro.poutine.trace() as tr:
        add_logistic_component(
            child_name="child1",
            child_continuous_parents=mock_data_cont,
            child_categorical_parents=mock_data_cat,
            leeway=0.5,
            data_plate=data_plate,
            categorical_levels=categorical_levels,
        )

    mean_prediction_child = tr.trace.nodes["mean_outcome_prediction_child1"]["value"]
    child_probs = tr.trace.nodes["child_probs_child1"]["value"]

    assert mean_prediction_child.shape == torch.Size([3])
    assert child_probs.shape == torch.Size([3])

    weights_categorical = {}
    for name in mock_data_cat.keys():
        weights_categorical[name] = tr.trace.nodes[
            f"weights_categorical_{name}_child1"
        ]["value"]

    categorical_contrib = torch.zeros(3)
    for name, tensor in mock_data_cat.items():
        categorical_contrib += weights_categorical[name][..., tensor]

    continuous_contrib = torch.zeros(3)
    bias = tr.trace.nodes["bias_continuous_child1"]["value"]
    for key, value in mock_data_cont.items():

        weight = tr.trace.nodes[f"weight_continuous_{key}_to_child1"]["value"]
        continuous_contrib += weight * value
    continuous_contrib += bias

    expected_mean_prediction = categorical_contrib + continuous_contrib

    expected_probs = torch.sigmoid(expected_mean_prediction)

    assert torch.allclose(child_probs, expected_probs, atol=1e-6)


def test_add_ratio_component():

    data_plate = pyro.plate("data_plate", 3)

    with pyro.poutine.trace() as tr:
        add_ratio_component(
            child_name="child1",
            child_continuous_parents=mock_data_cont,
            child_categorical_parents=mock_data_cat,
            leeway=0.5,
            data_plate=data_plate,
            categorical_levels=categorical_levels,
        )

    sigma_child = tr.trace.nodes["sigma_child1"]["value"]
    mean_prediction_child = tr.trace.nodes["mean_outcome_prediction_child1"]["value"]
    child_probs = tr.trace.nodes["child_probs_child1"]["value"]

    assert sigma_child.shape == torch.Size([])
    assert mean_prediction_child.shape == torch.Size([3])
    assert child_probs.shape == torch.Size([3])

    weights_categorical = {}
    for name in mock_data_cat.keys():
        weights_categorical[name] = tr.trace.nodes[
            f"weights_categorical_{name}_child1"
        ]["value"]

    categorical_contrib = torch.zeros(3)
    for name, tensor in mock_data_cat.items():
        categorical_contrib += weights_categorical[name][..., tensor]

    continuous_contrib = torch.zeros(3)
    bias = tr.trace.nodes["bias_continuous_child1"]["value"]
    for key, value in mock_data_cont.items():
        weight = tr.trace.nodes[f"weight_continuous_{key}_to_child1"]["value"]
        continuous_contrib += weight * value
    continuous_contrib += bias

    expected_mean_prediction = categorical_contrib + continuous_contrib

    expected_probs = torch.sigmoid(expected_mean_prediction)

    assert torch.allclose(child_probs, expected_probs, atol=1e-6)
