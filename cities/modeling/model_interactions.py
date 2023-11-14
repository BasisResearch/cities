import os

import dill
import pyro
import pyro.distributions as dist
import torch

from cities.modeling.modeling_utils import (
    prep_data_for_interaction_inference,
    train_interactions_model,
)
from cities.utils.cleaning_utils import find_repo_root


class InteractionsModel:
    def __init__(
        self,
        outcome_dataset,
        intervention_dataset,
        intervention_variable,
        forward_shift=2,
        num_iterations=1500,
        plot_loss=False,
    ):
        self.outcome_dataset = outcome_dataset
        self.intervention_dataset = intervention_dataset
        self.intervention_variable = intervention_variable
        self.forward_shift = forward_shift
        self.num_iterations = num_iterations
        self.plot_loss = plot_loss
        self.root = find_repo_root()

        data = prep_data_for_interaction_inference(
            outcome_dataset=self.outcome_dataset,
            intervention_dataset=self.intervention_dataset,
            intervention_variable=self.intervention_variable,
            forward_shift=2,
        )

        for key, value in data.items():
            setattr(self, key, value)

        self.model = cities_model_interactions

        self.model_args = (
            self.N_t,
            self.N_cov,
            self.N_s,
            self.N_u,
            self.N_obs,
            self.state_index_sparse,
            self.state_index,
            self.time_index,
            self.unit_index,
        )

        self.model_rendering = pyro.render_model(
            self.model, model_args=self.model_args, render_distributions=True
        )

        self.model_conditioned = pyro.condition(
            self.model,
            data={"T": self.intervention, "Y": self.y, "X": self.covariates_sparse},
        )

    def train_interactions_model(self):
        self.guide = train_interactions_model(
            model=self.model_conditioned,
            model_args=self.model_args,
            num_iterations=self.num_iterations,
            plot_loss=self.plot_loss,
        )

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

        print(f"Guide {guide_name} has been saved.")


def cities_model_interactions(
    N_t,
    N_cov,
    N_s,
    N_u,
    N_obs,
    state_index_sparse,
    state_index,
    time_index,
    unit_index,
    leeway=0.2,
):
    Y_bias = pyro.sample("Y_bias", dist.Normal(0, leeway))
    T_bias = pyro.sample("T_bias", dist.Normal(0, leeway))

    weight_TY = pyro.sample("weight_TY", dist.Normal(0, leeway))

    sigma_T = pyro.sample("sigma_T", dist.Exponential(1))
    sigma_Y = pyro.sample("sigma_Y", dist.Exponential(1))

    observations_plate = pyro.plate("observations_plate", N_obs, dim=-1)

    counties_plate = pyro.plate("counties_plate", N_u, dim=-2)
    states_plate = pyro.plate("states_plate", N_s, dim=-3)
    covariates_plate = pyro.plate("covariates_plate", N_cov, dim=-4)
    time_plate = pyro.plate("time_plate", N_t, dim=-5)

    with covariates_plate:
        X_bias = pyro.sample("X_bias", dist.Normal(0, leeway)).squeeze()
        sigma_X = pyro.sample("sigma_X", dist.Exponential(1)).squeeze()
        weight_XT = pyro.sample("weight_XT", dist.Normal(0, leeway)).squeeze()
        weight_XY = pyro.sample("weight_XY", dist.Normal(0, leeway)).squeeze()

    with states_plate:
        weight_UsT = pyro.sample("weight_UsT", dist.Normal(0, leeway)).squeeze()
        weight_UsY = pyro.sample("weight_UsY", dist.Normal(0, leeway)).squeeze()

        with covariates_plate:
            weight_UsX = pyro.sample("weight_UsX", dist.Normal(0, leeway)).squeeze()

    with time_plate:
        weight_UtT = pyro.sample("weight_UtT", dist.Normal(0, leeway)).squeeze()
        weight_UtY = pyro.sample("weight_UtY", dist.Normal(0, leeway)).squeeze()

    with counties_plate:
        UsX_weight_selected = weight_UsX.squeeze().T.squeeze()[state_index_sparse]
        X_means = torch.einsum("c,uc->uc", X_bias, UsX_weight_selected)
        X = pyro.sample("X", dist.Normal(X_means, sigma_X)).squeeze()

    XT_weighted = torch.einsum("uc, c -> u", X, weight_XT)
    XY_weighted = torch.einsum("uc, c -> u", X, weight_XY)

    with observations_plate:
        T_mean = (
            T_bias
            + weight_UtT[time_index]
            + weight_UsT[state_index]
            + XT_weighted[unit_index]
        )

        T = pyro.sample("T", dist.Normal(T_mean, sigma_T))

        Y_mean = (
            Y_bias
            + weight_UtY[time_index]
            + weight_UsY[state_index]
            + weight_TY * T
            + XY_weighted[unit_index]
        )

        Y = pyro.sample("Y", dist.Normal(Y_mean, sigma_Y))

    return Y
