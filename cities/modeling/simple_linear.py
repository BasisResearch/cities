import contextlib
from typing import Dict, Optional

import torch

import pyro
import pyro.distributions as dist

# TODO no major causal assumptions are added
# TODO add year/month latents
# TODO add neighborhood latents as impacting parcel areas and limits


def get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor]):
    N_categorical = len(categorical.keys())
    N_continuous = len(continuous.keys())

    if N_categorical > 0:
        n = len(next(iter(categorical.values())))
    elif N_continuous > 0:
        n = len(next(iter(continuous.values())))

    return N_categorical, N_continuous, n


class SimpleLinear(pyro.nn.PyroModule):
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

        # potentially move away from init as somewhat useless
        # for easy use of Predictive on other data

        self.leeway = leeway

        self.N_categorical, self.N_continuous, n = get_n(categorical, continuous)

        # you might need and pass further the original
        #  categorical levels of the training data
        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = dict()
            for name in categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])

    def forward(
        self,
        categorical: Dict[str, torch.Tensor],
        continuous: Dict[str, torch.Tensor],
        outcome: Optional[torch.Tensor] = None,
        categorical_levels: Optional[Dict[str, torch.Tensor]] = None,
        leeway=0.9,
    ):

        # can't take len of outcome
        # as in predictive check we might use outcome=None
        # can't assume either categorical or continuous is not empty

        N_categorical, N_continuous, n = get_n(categorical, continuous)

        if N_categorical == 0 and N_continuous == 0:
            raise ValueError("At least one input feature required.")

        categorical_contribution_outcome = torch.zeros(1, 1, 1, n)
        bias_continuous_outcome = torch.zeros(1, 1, 1, n)
        continuous_contribution_outcome = torch.zeros(1, 1, 1, n)

        sigma_outcome = pyro.sample("sigma", dist.Exponential(1.0))

        data_plate = pyro.plate("data", size=n, dim=-1)

        #################################################################################
        # add plates and linear contribution to outcome for categorical variables if any
        #################################################################################

        if N_categorical > 0:

            # Predictive and PredictiveModel don't seem to inherit much
            # of the self attributes, so we need to get them here
            # or grab the original ones from the model object passed to Predictive
            # while allowing them to be passed as arguments, as some
            # levels might be missing in new data for which we want to make predictions
            categorical_names = list(categorical.keys())
            if categorical_levels is None:
                categorical_levels = dict()
                for name in categorical_names:
                    categorical_levels[name] = torch.unique(categorical[name])

            weights_categorical_outcome = dict()
            objects_cat_weighted = {}

            for name in categorical_names:

                weights_categorical_outcome[name] = pyro.sample(
                    f"weights_categorical_{name}",
                    dist.Normal(0.0, self.leeway)
                    .expand(categorical_levels[name].shape)
                    .to_event(1),
                )

                objects_cat_weighted[name] = weights_categorical_outcome[name][
                    ..., categorical[name]
                ]

            values = list(objects_cat_weighted.values())
            for i in range(1, len(values)):
                values[i] = values[i].view(values[0].shape)

            categorical_contribution_outcome = torch.stack(
                values,
                dim=0,
            ).sum(dim=0)

        #################################################################################
        # add a plate and linear contribution to outcome for continuous variables if any
        #################################################################################

        if N_continuous > 0:

            continuous_stacked = torch.stack(list(continuous.values()), dim=0)

            bias_continuous_outcome = pyro.sample(
                "bias_continuous",
                dist.Normal(0.0, self.leeway)
                .expand([continuous_stacked.shape[-2]])
                .to_event(1),
            )

            weight_continuous_outcome = pyro.sample(
                "weight_continuous",
                dist.Normal(0.0, self.leeway)
                .expand([continuous_stacked.shape[-2]])
                .to_event(1),
            )

            continuous_contribution_outcome = (
                bias_continuous_outcome.sum()
                + torch.einsum(
                    "...cd, ...c -> ...d", continuous_stacked, weight_continuous_outcome
                )
            )

        #################################################################################
        # linear model for outcome
        #################################################################################

        with data_plate:

            mean_outcome_prediction = pyro.deterministic(
                "mean_outcome_prediction",
                categorical_contribution_outcome + continuous_contribution_outcome,
                event_dim=0,
            )

            outcome_observed = pyro.sample(
                "outcome_observed",
                dist.Normal(mean_outcome_prediction, sigma_outcome),
                obs=outcome,
            )

        return outcome_observed


@contextlib.contextmanager
def RegisterInput(model, kwargs):

    assert "categorical" in kwargs.keys()

    old_forward = model.forward

    def new_forward(**_kwargs):
        new_kwargs = _kwargs.copy()
        for key in _kwargs["categorical"].keys():
            new_kwargs["categorical"][key] = pyro.sample(
                key, dist.Delta(_kwargs["categorical"][key])
            )

        for key in _kwargs["continuous"].keys():
            new_kwargs["continuous"][key] = pyro.sample(
                key, dist.Delta(_kwargs["continuous"][key])
            )
        return old_forward(**new_kwargs)

    model.forward = new_forward
    yield
    model.forward = old_forward


# TODO rewrite input registration as more general function on model class

# class SimpleLinearRegisteredInput(pyro.nn.PyroModule):
#     def __init__(
#         self,
#         model,
#         categorical=Dict[str, torch.Tensor],
#         continuous=Dict[str, torch.Tensor],
#         outcome=None,
#         categorical_levels=None,
#     ):
#         super().__init__()
#         self.model = model

#         n = get_n(categorical, continuous)[2]

#         if categorical_levels is None:
#             categorical_levels = dict()
#             for name in categorical.keys():
#                 categorical_levels[name] = torch.unique(categorical[name])
#         self.categorical_levels = categorical_levels

#         def unconditioned_model():
#             _categorical = {}
#             _continuous = {}
#             with pyro.plate("initiate", size=n, dim=-8):
#                 for key in categorical.keys():
#                     _categorical[key] = pyro.sample(
#                         f"categorical_{key}", dist.Bernoulli(0.5)
#                     )
#                 for key in continuous.keys():
#                     _continuous[key] = pyro.sample(
#                         f"continuous_{key}", dist.Normal(0, 1)
#                     )
#             return self.model(
#                 categorical=_categorical,
#                 continuous=_continuous,
#                 outcome=None,
#                 categorical_levels=self.categorical_levels,
#             )

#         self.unconditioned_model = unconditioned_model

#         data = {
#             **{f"categorical_{key}": categorical[key] for key in categorical.keys()},
#             **{f"continuous_{key}": continuous[key] for key in continuous.keys()},
#         }

#         self.data = data

#         conditioned_model = condition(self.unconditioned_model, data=self.data)

#         self.conditioned_model = conditioned_model

#     def forward(self):
#         return self.conditioned_model()


# TODO mypy linting

# + mypy --ignore-missing-imports cities/
# cities/modeling/simple_linear.py:26: error: Name "pyro.nn.PyroModule" is not defined  [name-defined]
# cities/modeling/simple_linear.py:72: error: Module has no attribute "sample"  [attr-defined]
# cities/modeling/simple_linear.py:74: error: Module has no attribute "plate"  [attr-defined]
# cities/modeling/simple_linear.py:97: error: Module has no attribute "plate"  [attr-defined]
# cities/modeling/simple_linear.py:102: error: Module has no attribute "sample"  [attr-defined]
# cities/modeling/simple_linear.py:143: error: Module has no attribute "plate"  [attr-defined]
# cities/modeling/simple_linear.py:144: error: Module has no attribute "sample"  [attr-defined]
# cities/modeling/simple_linear.py:154: error: Module has no attribute "sample"  [attr-defined]
# cities/modeling/simple_linear.py:176: error: Module has no attribute "deterministic"  [attr-defined]
# cities/modeling/simple_linear.py:182: error: Module has no attribute "sample"  [attr-defined]
# cities/modeling/simple_linear.py:191: error: Name "pyro.nn.PyroModule" is not defined  [name-defined]
