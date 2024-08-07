import copy
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pyro
import seaborn as sns
import torch
from pyro.infer import Predictive
from torch.utils.data import DataLoader, random_split

from cities.modeling.svi_inference import run_svi_inference
from cities.utils.data_grabber import find_repo_root
from cities.utils.data_loader import select_from_data

root = find_repo_root()


def prep_data_for_test(
    data_path: Optional[str] = None, train_size: float = 0.8
) -> Tuple[DataLoader, DataLoader, list]:

    if data_path is None:
        data_path = os.path.join(root, "data/minneapolis/processed/zoning_dataset.pt")
    zoning_dataset_read = torch.load(data_path)

    train_size = int(train_size * len(zoning_dataset_read))
    test_size = len(zoning_dataset_read) - train_size

    train_dataset, test_dataset = random_split(
        zoning_dataset_read, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    categorical_levels = zoning_dataset_read.categorical_levels

    return train_loader, test_loader, categorical_levels


def recode_categorical(
    kwarg_names: Dict[str, Any], train_loader: DataLoader, test_loader: DataLoader
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:

    assert all(
        item in kwarg_names.keys() for item in ["categorical", "continuous", "outcome"]
    )

    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))

    _train_data = select_from_data(train_data, kwarg_names)
    _test_data = select_from_data(test_data, kwarg_names)

    ####################################################
    # eliminate test categories not in the training data
    ####################################################
    def apply_mask(data, mask):
        return {key: val[mask] for key, val in data.items()}

    mask = torch.ones(len(_test_data["outcome"]), dtype=torch.bool)
    for key, value in _test_data["categorical"].items():
        mask = mask * torch.isin(
            _test_data["categorical"][key], (_train_data["categorical"][key].unique())
        )

    _test_data["categorical"] = apply_mask(_test_data["categorical"], mask)
    _test_data["continuous"] = apply_mask(_test_data["continuous"], mask)
    _test_data["outcome"] = _test_data["outcome"][mask]

    for key in _test_data["categorical"].keys():
        assert _test_data["categorical"][key].shape[0] == mask.sum()
    for key in _test_data["continuous"].keys():
        assert _test_data["continuous"][key].shape[0] == mask.sum()

    # raise error if sum(mask) < .5 * len(test_data['outcome'])
    if sum(mask) < 0.5 * len(_test_data["outcome"]):
        raise ValueError(
            "Sampled test data has too many new categorical levels, consider decreasing train size"
        )

    # ####################################
    # recode categorical variables to have
    # no index gaps in the training data
    # ####################################

    mappings = {}
    for name in _train_data["categorical"].keys():
        unique_train = torch.unique(_train_data["categorical"][name])
        mappings[name] = {v.item(): i for i, v in enumerate(unique_train)}
        _train_data["categorical"][name] = torch.tensor(
            [mappings[name][x.item()] for x in _train_data["categorical"][name]]
        )
        _test_data["categorical"][name] = torch.tensor(
            [mappings[name][x.item()] for x in _test_data["categorical"][name]]
        )

    return _train_data, _test_data


def test_performance(
    model_or_class: Union[Callable[..., Any], Any],
    kwarg_names: Dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    categorical_levels: Dict[str, torch.Tensor],
    outcome_type: str = "outcome",
    outcome_name: str = "outcome",
    n_steps: int = 600,
    plot: bool = True,
    lim: Optional[Tuple[float, float]] = None,
    is_class: bool = True,
) -> Dict[str, float]:

    _train_data, _test_data = recode_categorical(kwarg_names, train_loader, test_loader)

    pyro.clear_param_store()

    ######################
    # train and test
    ######################

    if is_class:
        model = model_or_class(**_train_data)

    else:
        model = model_or_class

    guide = run_svi_inference(
        model, n_steps=n_steps, lr=0.01, verbose=True, **_train_data
    )

    predictive = Predictive(model, guide=guide, num_samples=1000)

    categorical_levels = model.categorical_levels

    _train_data_for_preds = copy.deepcopy(_train_data)
    _test_data_for_preds = copy.deepcopy(_test_data)

    if outcome_type != "outcome":
        _train_data_for_preds[outcome_type][outcome_name] = None  # type: ignore
        _test_data_for_preds[outcome_type][outcome_name] = None  # type: ignore

    else:
        _train_data_for_preds[outcome_type] = None  # type: ignore

    samples_train = predictive(
        **_train_data_for_preds,
        categorical_levels=categorical_levels,
    )

    samples_test = predictive(
        **_test_data_for_preds,
        categorical_levels=categorical_levels,
    )

    train_predicted_mean = samples_train[outcome_name].squeeze().mean(dim=0)
    train_predicted_lower = samples_train[outcome_name].squeeze().quantile(0.05, dim=0)
    train_predicted_upper = samples_train[outcome_name].squeeze().quantile(0.95, dim=0)

    coverage_training = (
        _train_data[outcome_type][outcome_name]
        .squeeze()
        .gt(train_predicted_lower)
        .float()
        * _train_data[outcome_type][outcome_name]
        .squeeze()
        .lt(train_predicted_upper)
        .float()
    )

    null_residuals_train = (
        _train_data[outcome_type][outcome_name].squeeze()
        - _train_data[outcome_type][outcome_name].squeeze().mean()
    )

    null_mae_train = torch.abs(null_residuals_train).mean().item()

    residuals_train = (
        _train_data[outcome_type][outcome_name].squeeze() - train_predicted_mean
    )
    mae_train = torch.abs(residuals_train).mean().item()

    rsquared_train = (
        1
        - residuals_train.var()
        / _train_data[outcome_type][outcome_name].squeeze().var()
    )

    test_predicted_mean = samples_test[outcome_name].squeeze().mean(dim=0)
    test_predicted_lower = samples_test[outcome_name].squeeze().quantile(0.05, dim=0)
    test_predicted_upper = samples_test[outcome_name].squeeze().quantile(0.95, dim=0)

    coverage_test = (
        _test_data[outcome_type][outcome_name]
        .squeeze()
        .gt(test_predicted_lower)
        .float()
        * _test_data[outcome_type][outcome_name]
        .squeeze()
        .lt(test_predicted_upper)
        .float()
    )

    null_residuals_test = (
        _test_data[outcome_type][outcome_name].squeeze()
        - _test_data[outcome_type][outcome_name].squeeze().mean()
    )

    null_mae_test = torch.abs(null_residuals_test).mean().item()

    residuals_test = (
        _test_data[outcome_type][outcome_name].squeeze() - test_predicted_mean
    )
    mae_test = torch.abs(residuals_test).mean().item()

    rsquared_test = (
        1
        - residuals_test.var() / _test_data[outcome_type][outcome_name].squeeze().var()
    )

    print(rsquared_train, rsquared_test)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].scatter(
            x=_train_data[outcome_type][outcome_name],
            y=train_predicted_mean,
            s=6,
            alpha=0.5,
        )
        axs[0, 0].set_title(
            "Training data, ratio of outcomes within 95% CI: {:.2f}".format(
                coverage_training.mean().item()
            )
        )

        if lim is not None:
            axs[0, 0].set_xlim(lim)
            axs[0, 0].set_ylim(lim)
        axs[0, 0].set_xlabel("observed values")
        axs[0, 0].set_ylabel("mean predicted values")

        axs[0, 1].hist(residuals_train, bins=50)

        axs[0, 1].set_title(
            "Training set residuals, MAE (null): {:.2f} ({:.2f}), Rsquared: {:.2f}".format(
                mae_train, null_mae_train, rsquared_train.item()
            )
        )
        axs[0, 1].set_xlabel("residuals")
        axs[0, 1].set_ylabel("frequency")

        axs[1, 0].scatter(
            x=_test_data[outcome_type][outcome_name],
            y=test_predicted_mean,
            s=6,
            alpha=0.5,
        )
        axs[1, 0].set_title(
            "Test data, ratio of outcomes within 95% CI: {:.2f}".format(
                coverage_test.mean().item()
            )
        )
        axs[1, 0].set_xlabel("true values")
        axs[1, 0].set_ylabel("mean predicted values")
        if lim is not None:
            axs[1, 0].set_xlim(lim)
            axs[1, 0].set_ylim(lim)

        axs[1, 1].hist(residuals_test, bins=50)

        axs[1, 1].set_title(
            "Test set residuals, MAE (null): {:.2f} ({:.2f}), Rsquared: {:.2f}".format(
                mae_test, null_mae_test, rsquared_test.item()
            )
        )

        axs[1, 1].set_xlabel("residuals")
        axs[1, 1].set_ylabel("frequency")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        sns.despine()

        fig.suptitle("Model evaluation", fontsize=16)

        plt.show()

    return {
        "mae_null_train": null_mae_train,
        "mae_null_test": null_mae_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "rsquared_train": rsquared_train,
        "rsquared_test": rsquared_test,
        "coverage_train": coverage_training.mean().item(),
        "coverage_test": coverage_test.mean().item(),
    }
