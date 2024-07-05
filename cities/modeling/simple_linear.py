import contextlib
from typing import Dict, List, Optional

import pyro
import pyro.distributions as dist
import torch

# TODO no major causal assumptions are incorporated


def get_n(categorical: Dict[str, torch.Tensor], continuous: Dict[str, torch.Tensor]):
    N_categorical = len(categorical.keys())
    N_continuous = len(continuous.keys())

    if N_categorical > 0:
        n = len(next(iter(categorical.values())))
    elif N_continuous > 0:
        n = len(next(iter(continuous.values())))

    return N_categorical, N_continuous, n


class SimpleLinear(pyro.nn.PyroModule):  # type: ignore
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

        # can't take len of outcome
        # as in predictive check we might use outcome=None
        # can't assume either categorical or continuous is not empty

        N_categorical, N_continuous, n = get_n(categorical, continuous)

        if N_categorical == 0 and N_continuous == 0:
            raise ValueError("At least one input feature required.")

        categorical_contribution_outcome = torch.zeros(1, 1, 1, n)
        bias_continuous_outcome = torch.zeros(1, 1, 1, n)
        continuous_contribution_outcome = torch.zeros(1, 1, 1, n)

        # TODO figure out why mypy complains
        sigma_outcome = pyro.sample("sigma", dist.Exponential(1.0))  # type: ignore

        data_plate = pyro.plate("data", size=n, dim=-1)  # type: ignore

        #################################################################################
        # add plates and linear contribution to outcome for categorical variables if any
        #################################################################################

        if N_categorical > 0:

            # Predictive and PredictiveModel don't seem to inherit much
            # of the self attributes, so we need to get them here
            # or grab the original ones from the model object passed to Predictive
            # while allowing them to be passed as arguments, as some
            # levels might be missing in new data for which we want to make predictions
            # or in the training/test split
            categorical_names = list(categorical.keys())
            if categorical_levels is None:
                categorical_levels = dict()
                for name in categorical_names:
                    categorical_levels[name] = torch.unique(categorical[name])

            weights_categorical_outcome = dict()
            objects_cat_weighted = {}

            for name in categorical_names:

                weights_categorical_outcome[name] = pyro.sample(  # type: ignore
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

            bias_continuous_outcome = pyro.sample(  # type: ignore
                "bias_continuous",
                dist.Normal(0.0, self.leeway)
                .expand([continuous_stacked.shape[-2]])
                .to_event(1),
            )

            weight_continuous_outcome = pyro.sample(  # type: ignore
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

            mean_outcome_prediction = pyro.deterministic(  # type: ignore
                "mean_outcome_prediction",
                categorical_contribution_outcome + continuous_contribution_outcome,
                event_dim=0,
            )

            outcome_observed = pyro.sample(  # type: ignore
                "outcome_observed",
                dist.Normal(mean_outcome_prediction, sigma_outcome),
                obs=outcome,
            )

        return outcome_observed


@contextlib.contextmanager
def RegisterInput(
    model, new_kwargs: Dict[str, List[str]]
):  # TODO mypy: can't use Callable as type hint no attribute forward

    assert "categorical" in new_kwargs.keys()

    old_forward = model.forward

    def new_forward(**kwargs):
        _kwargs = kwargs.copy()
        sampled_flags = (
            dict()
        )  # in some contexts multiple passes through the model result in multiple sample sites

        if "categorical" in new_kwargs.keys():
            for key in new_kwargs["categorical"].keys():
                if key not in sampled_flags:
                    _kwargs["categorical"][key] = pyro.sample(
                        key, dist.Delta(new_kwargs["categorical"][key])
                    )
                    sampled_flags[key] = True

        if "continuous" in new_kwargs.keys():
            for key in new_kwargs["continuous"].keys():
                if key not in sampled_flags:
                    _kwargs["continuous"][key] = pyro.sample(
                        key, dist.Delta(new_kwargs["continuous"][key])
                    )
                    sampled_flags[key] = True

        return old_forward(**_kwargs)

    model.forward = new_forward
    yield
    model.forward = old_forward
