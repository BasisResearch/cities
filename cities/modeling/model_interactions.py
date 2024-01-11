import logging
import os
from typing import Optional

import dill
import pyro
import pyro.distributions as dist
import torch

from cities.modeling.modeling_utils import (
    prep_wide_data_for_inference,
    train_interactions_model,
)
from cities.utils.data_grabber import DataGrabber, find_repo_root


class InteractionsModel:
    def __init__(
        self,
        outcome_dataset: str,
        intervention_dataset: str,
        intervention_variable: Optional[str] = None,
        forward_shift: int = 2,
        num_iterations: int = 1500,
        num_samples: int = 1000,
        plot_loss: bool = False,
    ):
        self.outcome_dataset = outcome_dataset
        self.intervention_dataset = intervention_dataset
        self.forward_shift = forward_shift
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.plot_loss = plot_loss
        self.root = find_repo_root()

        if intervention_variable:
            self.intervention_variable = intervention_variable
        else:
            _dg = DataGrabber()
            _dg.get_features_std_long([intervention_dataset])
            self.intervention_variable = _dg.std_long[intervention_dataset].columns[-1]

        self.data = prep_wide_data_for_inference(
            outcome_dataset=self.outcome_dataset,
            intervention_dataset=self.intervention_dataset,
            forward_shift=self.forward_shift,
        )

        self.model = model_cities_interaction

        self.model_args = self.data["model_args"]

        self.model_conditioned = pyro.condition(
            self.model,
            data={"T": self.data["t"], "Y": self.data["y"], "X": self.data["x"]},
        )

        self.model_rendering = pyro.render_model(
            self.model, model_args=self.model_args, render_distributions=True
        )

    def train_interactions_model(self):
        self.guide = train_interactions_model(
            conditioned_model=self.model_conditioned,
            model_args=self.model_args,
            num_iterations=self.num_iterations,
            plot_loss=self.plot_loss,
        )

    def sample_from_guide(self):
        predictive = pyro.infer.Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=self.num_samples,
            parallel=False,
        )
        self.samples = predictive(*self.model_args)

    def save_guide(self):
        guide_name = (
            f"{self.intervention_dataset}_{self.outcome_dataset}_{self.forward_shift}"
        )
        serialized_guide = dill.dumps(self.guide)
        file_path = os.path.join(
            self.root, "data/model_guides", f"{guide_name}_guide.pkl"
        )
        with open(file_path, "wb") as file:
            file.write(serialized_guide)
        param_path = os.path.join(
            self.root, "data/model_guides", f"{guide_name}_params.pth"
        )
        pyro.get_param_store().save(param_path)

        logging.info(
            f"Guide and params for {self.intervention_dataset}",
            f"{self.outcome_dataset} with shift {self.forward_shift}",
            "has been saved.",
        )


def model_cities_interaction(
    N_t,
    N_cov,
    N_s,
    N_u,
    state_index,
    unit_index,
    leeway=0.9,
):
    bias_Y = pyro.sample("bias_Y", dist.Normal(0, leeway))
    bias_T = pyro.sample("bias_T", dist.Normal(0, leeway))

    weight_TY = pyro.sample("weight_TY", dist.Normal(0, leeway))

    sigma_T = pyro.sample("sigma_T", dist.Exponential(1))
    sigma_Y = pyro.sample("sigma_Y", dist.Exponential(1))

    counties_plate = pyro.plate("counties_plate", N_u, dim=-1)
    states_plate = pyro.plate("states_plate", N_s, dim=-2)
    covariates_plate = pyro.plate("covariates_plate", N_cov, dim=-3)
    time_plate = pyro.plate("time_plate", N_t, dim=-4)

    with covariates_plate:
        bias_X = pyro.sample("bias_X", dist.Normal(0, leeway))
        sigma_X = pyro.sample("sigma_X", dist.Exponential(1))
        weight_XT = pyro.sample("weight_XT", dist.Normal(0, leeway))
        weight_XY = pyro.sample("weight_XY", dist.Normal(0, leeway))

    with states_plate:
        bias_stateT = pyro.sample("bias_stateT", dist.Normal(0, leeway))
        bias_stateY = pyro.sample("bias_stateY", dist.Normal(0, leeway))

        with covariates_plate:
            bias_stateX = pyro.sample("bias_stateX", dist.Normal(0, leeway))

    with time_plate:
        bias_timeT = pyro.sample("bias_timeT", dist.Normal(0, leeway))
        bias_timeY = pyro.sample("bias_timeY", dist.Normal(0, leeway))

    with counties_plate:
        with covariates_plate:
            mean_X = pyro.deterministic(
                "mean_X",
                torch.einsum(
                    "...xdd,...xcd->...xdc", bias_X, bias_stateX[..., state_index, :]
                ),
            )

            X = pyro.sample("X", dist.Normal(mean_X[..., unit_index], sigma_X))

            XT_weighted = torch.einsum(
                "...xdc, ...xdd -> ...dc", X, weight_XT
            ).unsqueeze(-2)
            XY_weighted = torch.einsum(
                "...xdc, ...xdd -> ...dc", X, weight_XY
            ).unsqueeze(-2)

        with time_plate:
            bias_stateT_tiled = pyro.deterministic(
                "bias_stateT_tiled",
                torch.einsum("...cd -> ...dc", bias_stateT[..., state_index, :]),
            )

            mean_T = pyro.deterministic(
                "mean_T", bias_T + bias_timeT + bias_stateT_tiled + XT_weighted
            )

            T = pyro.sample("T", dist.Normal(mean_T, sigma_T))

            bias_stateY_tiled = pyro.deterministic(
                "bias_stateY_tiled",
                torch.einsum("...cd -> ...dc", bias_stateY[..., state_index, :]),
            )

            mean_Y = pyro.deterministic(
                "mean_Y",
                bias_Y + bias_timeY + bias_stateY_tiled + XY_weighted + weight_TY * T,
            )
            Y = pyro.sample("Y", dist.Normal(mean_Y, sigma_Y))

    return Y
