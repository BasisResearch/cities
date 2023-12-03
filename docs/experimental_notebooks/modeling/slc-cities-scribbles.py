import numpy as np
from scipy import stats

import torch
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoDelta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from chirho.indexed.handlers import IndexPlatesMessenger
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.indexed.ops import IndexSet, gather, indices_of
from chirho.interventional.handlers import do
from chirho.observational.handlers import condition


from cities.utils.data_grabber import (DataGrabber, list_available_features, list_tensed_features)
from cities.utils.cleaning_utils import check_if_tensed 

from pyro.infer import Predictive


# get tensors for modeling

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#y = torch.linspace(10,11, steps = 100, dtype=torch.float32, device=device) 
#print("y: ", y)


intervention = torch.rand(100, dtype=torch.float32, device=device)
y = 100 + intervention * 20
#print("intervention: ", intervention)
#print("y: ", y) 
assert len(intervention) == len(y)


unit_index = torch.arange(0,10, dtype=torch.int, device=device).repeat(10)
#print("unit_index: ", unit_index)
assert len(unit_index) == len(y)

state_index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
                           dtype=torch.int, device=device).repeat(10)
#print("state_index: ", state_index)
assert len(state_index) == len(unit_index)

time_index = torch.arange(0,5, dtype=torch.int, device=device).repeat(20)
#print("time_index: ", time_index)   
assert len(time_index) == len(unit_index)


covariates_sparse = torch.rand((10,2), dtype = torch.float32, device=device)
unit_index_sparse = torch.arange(0,10, dtype = int, device=device) 
state_index_sparse = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

N_cov = covariates_sparse.shape[1] #number of covariates
N_u = covariates_sparse.shape[0] #number of units (counties)
N_obs = len(y) #number of observations
N_t = len(time_index.unique()) #number of time points
N_s = len(state_index.unique()) #number of states

assert covariates_sparse.shape[1] == covariates_sparse.shape[1]
assert len(unit_index_sparse) == N_u


# SETUP ENDS________________________________

# MODEL STARTS________________________________

def cities_model_E(N_t, N_s, state_index, time_index,
                    observed_intervention, observed_covariates_sparse,
                observed_y = None):
    
    
    T_bias = pyro.sample("T_bias", dist.Normal(0, 1))
    Y_bias = pyro.sample("Y_bias", dist.Normal(0, 1))
    
    weight_TY = pyro.sample("weight_TY", dist.Normal(0, .5))
    
    observations_plate = pyro.plate("observations_plate", N_obs, dim = -1)
    counties_plate = pyro.plate("counties_plate", N_u, dim = -2)
    states_plate = pyro.plate("states_plate", N_s, dim=-3)
    covariates_plate = pyro.plate("covariates_plate", N_cov, dim=-4)
    time_plate = pyro.plate("time_plate", N_t, dim=-5)

    with time_plate:
        # time-related confounders
        Ut = pyro.sample("Ut", dist.Normal(0, .4))
        
        # impact thereof on treatment
        weight_UtT = pyro.sample("weight_UtT", dist.Normal(0, .4))
        
        # impact thereof on outcome
        weight_UtY = pyro.sample("weight_UtY", dist.Normal(0, .4))
        
    
    with states_plate:
        # state-related confounders
        Us = pyro.sample("Us", dist.Normal(0, .4))
        
        # impact thereof on treatment
        weight_UsT = pyro.sample("weight_UsT", dist.Normal(0, .4))
        
        # impact thereof on outcome
        weight_UsY = pyro.sample("weight_UsY", dist.Normal(0, .4))
   
        with covariates_plate:
            weight_UsX = pyro.sample("weight_UsX", dist.Normal(0, .4))                    
            
    with covariates_plate:
        X_bias = pyro.sample("X_bias", dist.Normal(0, 1))
        weight_XT = pyro.sample("weight_XT", dist.Normal(0, .4))
        weight_XY = pyro.sample("weight_XY", dist.Normal(0, .4))
        
        
    with counties_plate:
        UsX_weighted = torch.einsum("ij...,j...->ij", weight_UsX, Us)
        UsX_weighted_selected = UsX_weighted[...,state_index_sparse].squeeze()
        X_means = torch.einsum("i...,ik->ik" ,X_bias, UsX_weighted_selected)[...,unit_index_sparse].T
        X = pyro.sample("X", dist.Normal(X_means, .4))
        
 
    
    with observations_plate:
      
        UtT_weighted = torch.einsum("...i,...j->...ij", weight_UtT, Ut)
        print("UtT_weighted: ", UtT_weighted.shape)
     
        UsT_weighted = torch.einsum("...i,...j->...ij", weight_UsT, Us)
        print("UsT_weighted: ", UsT_weighted.shape)
        
        XT_weighted = torch.einsum("i...,ji->j", weight_XT, X)
        print("XT_weighted", XT_weighted.shape)
        
        T_mean = (T_bias + UtT_weighted[time_index, ...].squeeze() 
                  + UsT_weighted[state_index, ...].squeeze() + XT_weighted[unit_index])
        print("T_mean",T_mean.shape)
        
        # TODO add prior over this variance
        T = pyro.sample("T", dist.Normal(T_mean, .4))
   
        UtY_weighted = torch.einsum("...i,...j->...ij", weight_UtY, Ut)
        UsY_weighted = torch.einsum("...i,...j->...ij", weight_UsY, Us)
        
        XY_weighted = torch.einsum("i...,ji->j", weight_XY, X)
        
        Y_mean = (Y_bias + UtY_weighted[time_index, ...].squeeze() +
                UsY_weighted[state_index, ...].squeeze() +
                weight_TY * T + XY_weighted[unit_index]
                )   
        
        # TODO add prior over this variance
        Y = pyro.sample("Y", dist.Normal(Y_mean, .4))
   
   
        
    
    
    
with pyro.poutine.trace() as tr:
    cities_model_E(N_t, N_s, state_index, time_index,
                    intervention, covariates_sparse, observed_y = y)

pyro.render_model(cities_model_E, 
                  model_args=(N_t, N_s, state_index, time_index,
                               intervention, covariates_sparse),
                  render_distributions=True, filename="docs/experimental_notebooks/cities_model_E.png")


for key in tr.trace.nodes.keys():
    if not key.endswith("_plate"):
        print(key, tr.trace.nodes[key]["value"].shape)
        
assert tr.trace.nodes["T"]["value"].shape == intervention.shape
assert tr,trace.nodes["Y"]["value"].shape == y.shape
assert tr.trace.nodes["X"]["value"].shape == covariates_sparse.shape