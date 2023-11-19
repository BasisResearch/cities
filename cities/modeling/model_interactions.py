import os
import logging

import dill
import pyro
import pyro.distributions as dist
import torch

from cities.utils.data_grabber import DataGrabber
from cities.modeling.modeling_utils import (prep_wide_data_for_inference, train_interactions_model)
from cities.utils.cleaning_utils import find_repo_root


class InteractionsModel:
    def __init__(
        self,
        outcome_dataset,
        intervention_dataset,
        intervention_variable = None,
        forward_shift=2,
        num_iterations=1500,
        num_samples = 1000,
        plot_loss=False,
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

        self.model_args = self.data['model_args']


        self.model = model_cities_interaction

        self.model_conditioned =   pyro.condition(
            self.model,
            data={"T": self.data['t'], "Y": self.data['y'], 
                  "X": self.data['x']},
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
        predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=self.num_samples, parallel=False)
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

        logging.info(f"Guide {guide_name} has been saved.")




def model_cities_interaction(
    N_t,
    N_cov,
    N_s,
    N_u,
    state_index,
    unit_index,
    leeway= .9,
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
        
            mean_X = pyro.deterministic("mean_X", torch.einsum("...xdd,...xcd->...xdc", bias_X, bias_stateX[...,state_index,:])) 
            
            X = pyro.sample("X", dist.Normal(mean_X[...,unit_index], sigma_X)) 
        
            XT_weighted =  torch.einsum("...xdc, ...xdd -> ...dc", X, weight_XT).unsqueeze(-2)
            XY_weighted =  torch.einsum("...xdc, ...xdd -> ...dc", X, weight_XY).unsqueeze(-2)

        with time_plate:
            
            bias_stateT_tiled = pyro.deterministic("bias_stateT_tiled", torch.einsum("...cd -> ...dc", bias_stateT[...,state_index,:]))
            
            mean_T = pyro.deterministic("mean_T",  bias_T + bias_timeT + bias_stateT_tiled +  XT_weighted)

            T = pyro.sample("T", dist.Normal(mean_T, sigma_T))
            
            bias_stateY_tiled = pyro.deterministic("bias_stateY_tiled", torch.einsum("...cd -> ...dc", 
                                                                        bias_stateY[...,state_index,:]))

            mean_Y = pyro.deterministic("mean_Y", 
                                        bias_Y + bias_timeY + bias_stateY_tiled + XY_weighted + weight_TY * T)
            Y = pyro.sample("Y", dist.Normal(mean_Y, sigma_Y))

    return Y












# def cities_model_interactions(
#     N_t,
#     N_cov,
#     N_s,
#     N_u,
#     N_obs,
#     state_index_sparse,
#     state_index,
#     time_index,
#     unit_index,
#     leeway=0.2,
# ):
#     Y_bias = pyro.sample("Y_bias", dist.Normal(0, leeway))
#     T_bias = pyro.sample("T_bias", dist.Normal(0, leeway))

#     weight_TY = pyro.sample("weight_TY", dist.Normal(0, leeway))

#     sigma_T = pyro.sample("sigma_T", dist.Exponential(1))
#     sigma_Y = pyro.sample("sigma_Y", dist.Exponential(1))

#     observations_plate = pyro.plate("observations_plate", N_obs, dim=-1)

#     counties_plate = pyro.plate("counties_plate", N_u, dim=-2)
#     states_plate = pyro.plate("states_plate", N_s, dim=-3)
#     covariates_plate = pyro.plate("covariates_plate", N_cov, dim=-4)
#     time_plate = pyro.plate("time_plate", N_t, dim=-5)

#     with covariates_plate:
#         X_bias = pyro.sample("X_bias", dist.Normal(0, leeway)).squeeze()
#         sigma_X = pyro.sample("sigma_X", dist.Exponential(1)).squeeze()
#         weight_XT = pyro.sample("weight_XT", dist.Normal(0, leeway)).squeeze()
#         weight_XY = pyro.sample("weight_XY", dist.Normal(0, leeway)).squeeze()

#     with states_plate:
#         weight_UsT = pyro.sample("weight_UsT", dist.Normal(0, leeway)).squeeze()
#         weight_UsY = pyro.sample("weight_UsY", dist.Normal(0, leeway)).squeeze()

#         with covariates_plate:
#             weight_UsX = pyro.sample("weight_UsX", dist.Normal(0, leeway)).squeeze()

#     with time_plate:
#         weight_UtT = pyro.sample("weight_UtT", dist.Normal(0, leeway)).squeeze()
#         weight_UtY = pyro.sample("weight_UtY", dist.Normal(0, leeway)).squeeze()

#     with counties_plate:
#         UsX_weight_selected = weight_UsX.squeeze().T.squeeze()[state_index_sparse]
#         X_means = torch.einsum("c,uc->uc", X_bias, UsX_weight_selected)
#         X = pyro.sample("X", dist.Normal(X_means, sigma_X)).squeeze()

#     XT_weighted = torch.einsum("uc, c -> u", X, weight_XT)
#     XY_weighted = torch.einsum("uc, c -> u", X, weight_XY)

#     with observations_plate:
#         T_mean = (
#             T_bias
#             + weight_UtT[time_index]
#             + weight_UsT[state_index]
#             + XT_weighted[unit_index]
#         )

#         T = pyro.sample("T", dist.Normal(T_mean, sigma_T))

#         Y_mean = (
#             Y_bias
#             + weight_UtY[time_index]
#             + weight_UsY[state_index]
#             + weight_TY * T
#             + XY_weighted[unit_index]
#         )

#         Y = pyro.sample("Y", dist.Normal(Y_mean, sigma_Y))

#     return Y
