import pyro
import torch
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist
from torch import nn
from chirho.dynamical.ops import State, simulate
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.handlers import LogTrajectory, StaticBatchObservation
import matplotlib.pyplot as plt
from torch import tensor as tnsr
from torch import Tensor as Tnsr
from typing import Any, Optional, Callable, List
import math
from sklearn.preprocessing import PolynomialFeatures

# https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
class LatentODEfunc(PyroModule):

    def __init__(self, latent_dim, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out



# class PolynomialBasis:
#     def __init__(self, degree: int, include_bias: bool = True) -> None:
#         assert degree >= 1
#         self.degree = degree
#         self.include_bias = include_bias
    
#     def __call__(self, x) -> torch.Tensor:
#         # FIXME this doesn't remove duplicate interaction terms for degree > 2
#         expanded = x
#         degree_d_term = x # base case
#         for d in range(self.degree - 1):
#             outer = torch.outer(x, degree_d_term)
#             indices = torch.triu_indices(outer.shape[0], outer.shape[1])
#             degree_d_term = outer[indices.unbind()]
#             expanded = torch.cat([expanded,  degree_d_term], axis=-1)
#         return expanded if not self.include_bias else torch.cat([tnsr([1.]), expanded], axis=-1)
    

# class SinBasis:
#     def __init__(self, offset: Tnsr, frequency: Tnsr):
#         self.offset = offset
#         self.frequency = frequency
    
#     def __call__(self, x: Tnsr) -> Tnsr:
#         return torch.sin(self.frequency[:, None] * (x[None, :] + self.offset[:, None])).flatten()


# class LinearDynamics(PyroModule):
#     def __init__(
#         self,
#         name: str,
#         example_state: State[Tnsr],
#         basis_fn,
#         weight_scale=1.,
#         link_fn: Optional[Callable]=None, 
#         mle: bool=False,
#     ):
#         self.name = name
#         super().__init__(name)
#         self.basis_fn = basis_fn
        
#         example_state = example_state.copy()
#         example_state['t'] = tnsr(0.0)  # because during execution time is added to the state.
#         n_basis_feats = self.basis_fn(flatten_state(example_state)).shape[-1]
#         self.weight = pyro.sample(
#             f"{self.name}_weights",
#             dist.Normal(0., weight_scale / math.sqrt(n_basis_feats)).expand((n_basis_feats,)).to_event(1))
#         self.link_fn = link_fn if link_fn is not None else lambda x: x
#         self.mle = mle

#         if self.mle:
#             self.weight = PyroParam(self.weight.clone().detach())

#     def __call__(self, x):
#         mean = self.basis_fn(x) @ self.weight
#         return self.link_fn(mean)
    

def flatten_state(state: State[torch.Tensor]):
    return torch.stack([state[k] for k in state.keys()], axis=-1)


class DollarGDPLatent(PyroModule):
    def __init__(self, expdec_param, dynamics_fn):
        super().__init__()
        self.expdec_param = expdec_param
        self.dynamics_fn = dynamics_fn

    @pyro.nn.pyro_method
    def diff(self, dstate: State[torch.Tensor], state: State[torch.Tensor]) -> None:

        # instant_dollars = self.annualized_dollars(flatten_state(state))
        # dstate["cumulative_dollars"] = instant_dollars

        # # Exponential decay of total dollars over time
        # dstate["cumulative_dollars_over_expdec_window"] = instant_dollars - self.expdec_param * state["cumulative_dollars_over_expdec_window"]

        # for i, latent_dynamic in enumerate(self.latents):
        #     Zi = state[f"Z{i}"]
        #     dstate[f"Z{i}"] = latent_dynamic(flatten_state(state)) #- 0.1 * torch.sign(Zi) * Zi ** 2.
        # dstate[f"Z{0}"] = self.latents[0](torch.atleast_1d(state['t']))

        dsdt = self.dynamics_fn(flatten_state(state))
        for i, Zi in enumerate(dsdt):
            dstate[f"Z{i}"] = Zi

    def forward(self, state: State[torch.Tensor]):
        dstate = State()
        self.diff(dstate, state)
        return dstate

    @pyro.nn.pyro_method
    def observation(self, X: State[torch.Tensor]) -> None:
        # # We don't observe the number of susceptible individuals directly, and instead can only infer it from the
        # #  number of test kits that are sold (which is a noisy function of the number of susceptible individuals).
        # event_dim = 1 if X["cont_propensity_score"].shape and X["cont_propensity_score"].shape[-1] > 1 else 0
        # instant_dollars = pyro.sample(f"instant_dollars", dist.Normal(X["cont_propensity_score"], 1.).to_event(event_dim)) 
        pass

#         GDP_{nt} ~ a instant_dollars_{nt} + b other stuff_{nt}
#
#         dlatent_GDP/dt = f_learnable_dynamics(instant_dollars, other covariate states)
#             GDP_t ~ N(latent_GDP_t, 1)
#
#
# Relate GDP to instant dollars at time and other time-varting covariates Xt
#
# dlatent_mean_fn/dt = some_function(instant_dollars_mean_t, covariates_mean_t)
# dinstant_dollars_mean/dt = some_function(covariates_mean_t)
# dcovariates_mean/dt = some_function()
#
# GDP ~ N(latent_mean_fn_t, 1)
# instant_dollars ~ N(instant_dollars_mean_t, 1)
#
# Xt = continuous time stoch proccess
# At = continuous time stoch proccess
# Yt = continuous time stoch proccess
#
# P(Yt | At, Xt, Y_lagged_t)
#
# f(At, Xt, Y_lagged_t)
#
#
# P(At+1 | At, Xt)
#
#
#
#
#
# X_t = values of the observed covariates / confounders at time t
# A_t = instant dollars at time t
# P(A_t | X_t) = Normal(f(X_t), 1)
# E[]
# cont_propensity_score = f(X_t)
#
# X = convariates
# A = treatment
#
# N(f(X), 1)
#
# A_true ~ f(X) + N(0, 1)
#
# GDP_later = f(A_true)
#
# A_noisy = A_true + N(0, 1)


NUM_LATENTS = 1

def target_fn(t):
    return torch.exp(-t/15.) * torch.cos(.4 * t)

initial_state = State(
    # cumulative_dollars=tnsr(0.0),
    # cumulative_dollars_over_expdec_window=tnsr(0.0),
    **{f"Z{i}": target_fn(tnsr(0.0))
       for i in range(NUM_LATENTS)}
)


def bayes_dynsys():
    
    # latents = torch.nn.ModuleList([
    #     LinearDynamics(
    #         name=f"Z{i}",
    #         # poly_basis_fn=PolynomialBasis(degree=2),
    #         # basis_fn=SinBasis(frequency=tnsr([1.0, 1.0, 1.0, 1.0]), offset=tnsr([0.0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi])),
    #         basis_fn=LatentODEfunc(latent_dim=NUM_LATENTS + 1),
    #         example_state=initial_state,
    #         weight_scale=1,
    #         mle=True
    #     ) for i in range(NUM_LATENTS)
    # ])

    dynsys = DollarGDPLatent(
        expdec_param=0.1,
        dynamics_fn=LatentODEfunc(latent_dim=NUM_LATENTS + 1)
    )
    
    return dynsys


def simulated_bayes_dynsys(dynsys: DollarGDPLatent, init_state, start_time, logging_times) -> State[torch.Tensor]:

    with LogTrajectory(logging_times) as lt:
        simulate(dynsys, init_state, start_time, logging_times[-1] + 1e-3, solver=TorchDiffEq())
    
    trajectory = lt.trajectory
    
    # # This is a small trick to make the solution variables available to pyro
    # [pyro.deterministic(k, trajectory[k]) for k in trajectory.keys()]

    return trajectory

start_time = tnsr(0.0)
end_time = torch.tensor(30.)
step_size = torch.tensor(0.1)
trajectory_times = torch.arange(start_time+step_size, end_time, step_size)

target_curve = target_fn(trajectory_times)

def loss_fn(model):
    traj = simulated_bayes_dynsys(model, initial_state, start_time, trajectory_times)
    # FIXME LogTrajectory seems to result in a trajectory of length one fewer than the requested trajectory times.
    return ((traj['Z0'] - target_curve) ** 2.).sum()


def fit_dynsys():

    model = bayes_dynsys()
    loss_fn(model) # compute the loss once to initialize any lazy parameters.

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 1000

    for i in range(num_steps):
        loss = loss_fn(model)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 1 == 0:
            print(f"loss: {loss:>7f}  [{i:>5d}/{num_steps:>5d}]")
    return model


model = fit_dynsys()

trajectory = simulated_bayes_dynsys(
    dynsys=model,
    init_state=initial_state,
    start_time=start_time,
    logging_times=trajectory_times
)

# plt.plot(trajectory["cumulative_dollars_over_expdec_window"], label="cumulative_dollars_over_expdec_window")
# plt.plot(trajectory["cumulative_dollars"], label="cumulative_dollars")
for i in range(1, NUM_LATENTS):
    plt.plot(trajectory[f"Z{i}"].detach(), label=f"Z{i}", linewidth=0.1)
plt.plot(trajectory['Z0'].detach(), label="Z0", linewidth=1.0)
plt.plot(target_curve, label="target_curve", linewidth=0.5, linestyle="--")

plt.show()
