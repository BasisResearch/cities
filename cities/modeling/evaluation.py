import os

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


def prep_data_for_test(train_size=0.8):
    zoning_data_path = os.path.join(
        root, "data/minneapolis/processed/zoning_dataset.pt"
    )
    zoning_dataset_read = torch.load(zoning_data_path)

    train_size = int(train_size * len(zoning_dataset_read))
    test_size = len(zoning_dataset_read) - train_size

    train_dataset, test_dataset = random_split(
        zoning_dataset_read, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    categorical_levels = zoning_dataset_read.categorical_levels

    return train_loader, test_loader, categorical_levels


def recode_categorical(kwarg_names, train_loader, test_loader):
    
    assert all(
        item in kwarg_names.keys() for item in ["categorical", "continuous", "outcome"]
    )
    assert kwarg_names["outcome"] not in kwarg_names["continuous"]

    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))

    _train_data = select_from_data(train_data, kwarg_names)
    _test_data = select_from_data(test_data, kwarg_names)

    #####################################################
    # eliminate test categories not in the training data
    #####################################################
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

    ######################################
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

    #for key in _train_data['categorical'].keys():
    #     print(key)
    #     print(_train_data['categorical'][key].unique())
    #     print(_test_data['categorical'][key].unique())
    #  TODO codsider adding assertion        
    #   assert torch.all(test_data['categorical'][key].unique() in _train_data['categorical'][key].unique())
    

    return _train_data, _test_data



def test_performance(
    model_or_class,
    kwarg_names,
    train_loader,
    test_loader,
    categorical_levels,
    n_steps=600,
    plot=True,
    is_class = True    
):
    _train_data, _test_data = recode_categorical(kwarg_names, 
                                                train_loader, test_loader)

    pyro.clear_param_store()
    # TODO perhaps remove the original categorical levels here

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

    samples_training = predictive(
        categorical=_train_data["categorical"],
        continuous=_train_data["continuous"],
        outcome=None,
        categorical_levels=categorical_levels,
    )

    samples_test = predictive(
        categorical=_test_data["categorical"],
        continuous=_test_data["continuous"],
        outcome=None,
        categorical_levels=categorical_levels,
    )

    train_predicted_mean = samples_training[kwarg_names['outcome']].squeeze().mean(dim=0)
    train_predicted_lower = (
        samples_training[kwarg_names['outcome']].squeeze().quantile(0.05, dim=0)
    )
    train_predicted_upper = (
        samples_training[kwarg_names['outcome']].squeeze().quantile(0.95, dim=0)
    )

    coverage_training = (
        _train_data["outcome"].squeeze().gt(train_predicted_lower).float()
        * _train_data["outcome"].squeeze().lt(train_predicted_upper).float()
    )
    residuals_train = _train_data["outcome"].squeeze() - train_predicted_mean
    mae_train = torch.abs(residuals_train).mean().item()

    rsquared_train = 1 - residuals_train.var() / _train_data["outcome"].squeeze().var()

    test_predicted_mean = samples_test[kwarg_names['outcome']].squeeze().mean(dim=0)
    test_predicted_lower = (
        samples_test[kwarg_names['outcome']].squeeze().quantile(0.05, dim=0)
    )
    test_predicted_upper = (
        samples_test[kwarg_names['outcome']].squeeze().quantile(0.95, dim=0)
    )

    coverage_test = (
        _test_data["outcome"].squeeze().gt(test_predicted_lower).float()
        * _test_data["outcome"].squeeze().lt(test_predicted_upper).float()
    )
    residuals_test = _test_data["outcome"].squeeze() - test_predicted_mean
    mae_test = torch.abs(residuals_test).mean().item()

    rsquared_test = 1 - residuals_test.var() / _test_data["outcome"].squeeze().var()

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].scatter(
            x=_train_data["outcome"], y=train_predicted_mean, s=6, alpha=0.5
        )
        axs[0, 0].set_title(
            "Training data, ratio of outcomes within 95% CI: {:.2f}".format(
                coverage_training.mean().item()
            )
        )
        axs[0, 0].set_xlabel("true outcome")
        axs[0, 0].set_ylabel("mean predicted outcome")

        axs[0, 1].hist(residuals_train, bins=50)
        axs[0, 1].set_title(
            "Training set residuals, Rsquared: {:.2f}".format(rsquared_train.item())
        )
        axs[0, 1].set_xlabel("residuals")
        axs[0, 1].set_ylabel("frequency")

        axs[1, 0].scatter(
            x=_test_data["outcome"], y=test_predicted_mean, s=6, alpha=0.5
        )
        axs[1, 0].set_title(
            "Test data, ratio of outcomes within 95% CI: {:.2f}".format(
                coverage_test.mean().item()
            )
        )
        axs[1, 0].set_xlabel("true outcome")
        axs[1, 0].set_ylabel("mean predicted outcome")

        axs[1, 1].hist(residuals_test, bins=50)
        axs[1, 1].set_title(
            "Test set residuals, Rsquared: {:.2f}".format(rsquared_test.item())
        )
        axs[1, 1].set_xlabel("residuals")
        axs[1, 1].set_ylabel("frequency")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        sns.despine()

        fig.suptitle("Model evaluation", fontsize=16)

        plt.show()

    return {
        "mae_train": mae_train,
        "mae_test": mae_test,
        "rsquared_train": rsquared_train,
        "rsquared_test": rsquared_test,
        "coverage_train": coverage_training.mean().item(),
        "coverage_test": coverage_test.mean().item(),
    }
