from typing import Dict, Optional

import torch

import pyro
import pyro.distributions as dist

# TODO no major causal assumptions are added
# TODO add year/month latents
# TODO add neighborhood latents as impacting parcel areas and limits


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

        # potentially move away from init as somewhat useless for easy use of Predictive on other data

        self.leeway = leeway

        self.N_categorical = len(categorical.keys())

        if self.N_categorical > 0 and categorical_levels is None:
            self.categorical_levels = dict()
            for name in categorical.keys():
                self.categorical_levels[name] = torch.unique(categorical[name])

        self.N_continuous = len(continuous.keys())

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
        N_categorical = len(categorical.keys())
        N_continuous = len(continuous.keys())

        if N_categorical == 0 and N_continuous == 0:
            raise ValueError("At least one input feature required.")

        if N_categorical > 0:
            n = len(next(iter(categorical.values())))
        elif N_continuous > 0:
            n = len(next(iter(continuous.values())))

        categorical_contribution_outcome = torch.zeros(1, 1, 1, n)
        bias_continuous_outcome = torch.zeros(1, 1, 1, n)
        continuous_contribution_outcome = torch.zeros(1, 1, 1, n)

        sigma_outcome = pyro.sample("sigma", dist.Exponential(1.0))

        data_plate = pyro.plate("data", size=n, dim=-1)

        running_dim = -2

        #################################################################################
        # add a plate and linear contribution to outcome for categorical variables if any
        #################################################################################

        if N_categorical > 0:

            # Predictive doesn't seem to inherit much of the self attributes, so we need to get them here
            # while allowing them to be passed as arguments, as some levels might be missing in data for
            # which we want to make predictions
            categorical_names = list(categorical.keys())
            categorical_levels = dict()
            for name in categorical_names:
                categorical_levels[name] = torch.unique(categorical[name])

            weights_categorical_outcome = dict()
            objects_cat_weighted = {}

            for name in categorical_names:
                with pyro.plate(
                    f"w_plate_{name}",
                    size=len(categorical_levels[name]),
                    dim=(running_dim),
                ):
                    weights_categorical_outcome[name] = pyro.sample(
                        f"weights_categorical_{name}", dist.Normal(0.0, self.leeway)
                    )
                running_dim -= 1

                while (
                    weights_categorical_outcome[name].shape[-1] == 1
                    and len(weights_categorical_outcome[name].shape) > 1
                ):
                    weights_categorical_outcome[name] = weights_categorical_outcome[
                        name
                    ].squeeze(-1)

                objects_cat_weighted[name] = weights_categorical_outcome[name][
                    ..., categorical[name]
                ]

            # most likely too add hoc and now redundant
            # max_shape_length = max([len(t.shape) for t in objects_cat_weighted.values()])
            # for name in categorical_names:
            #     while len(objects_cat_weighted[name].shape) < max_shape_length:
            #         objects_cat_weighted[name] = objects_cat_weighted[name].unsqueeze(0)

            values = list(objects_cat_weighted.values())
            for i in range(1,len(values)):
                values[i] = values[i].view(values[0].shape)

            categorical_contribution_outcome = torch.stack(
                values, dim=0
                #list(objects_cat_weighted.values()), dim=0
            ).sum(dim=0)

        #################################################################################
        # add a plate and linear contribution to outcome for continuous variables if any
        #################################################################################

        if N_continuous > 0:

            continuous_stacked = torch.stack(list(continuous.values()), dim=0)

            with pyro.plate("continuous", size=N_continuous, dim=running_dim):
                bias_continuous_outcome = pyro.sample(
                    "bias_continuous", dist.Normal(0.0, self.leeway)
                )

                while (
                    bias_continuous_outcome.shape[-1] == 1
                    and len(bias_continuous_outcome.shape) > 1
                ):
                    bias_continuous_outcome = bias_continuous_outcome.squeeze(-1)

                weight_continuous_outcome = pyro.sample(
                    "weight_continuous", dist.Normal(0.0, self.leeway)
                )
                while (
                    weight_continuous_outcome.shape[-1] == 1
                    and len(weight_continuous_outcome.shape) > 1
                ):
                    weight_continuous_outcome = weight_continuous_outcome.squeeze(-1)

            running_dim -= 1

            continuous_contribution_outcome = torch.einsum(
                "...d -> ...", bias_continuous_outcome
            ) + torch.einsum(
                "...cd, ...c -> ...d", continuous_stacked, weight_continuous_outcome
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
                dist.Normal(mean_outcome_prediction, sigma_outcome), obs = outcome
            )

        return outcome_observed
