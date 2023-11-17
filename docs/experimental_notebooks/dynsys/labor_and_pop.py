import pyro
import torch
from chirho.dynamical.ops import State, simulate
from typing import Dict, List, Tuple
import pyro.distributions as dist
from itertools import product
from torch import Tensor as Tnsr
from torch import tensor as tnsr
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.handlers import LogTrajectory, StaticBatchObservation, InterruptionEventLoop
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from chirho.observational.handlers import condition

pyro.settings.set(module_local_params=True)


def to_state_codes(fips_codes):
    return [fips // 1000 for fips in fips_codes]


# noinspection PyUnresolvedReferences
class FluxDynamics(pyro.nn.PyroModule):
    def __init__(self, fluxes: Tnsr, pairs: List[Tuple[str, str]], noise_scale: float):
        super().__init__()
        self.fluxes = fluxes  # .shape == (n_pairs, n_fips)
        self.pairs = pairs
        self.noise_scale = noise_scale

    @pyro.nn.pyro_method
    def diff(self, dstate: State[torch.Tensor], state: State[torch.Tensor]) -> None:

        # TODO should be a way to vectorize this too I think. Basically just reform the flux
        #  into a matrix with zeros on the diagonal, then matrix something something?
        for flux, (c1, c2) in zip(self.fluxes, self.pairs):

            # Note that flux here has shape (n_fips,), and that each state
            #  is a vector over all counties.
            dc1 = dstate.get(c1, torch.zeros_like(state[c1]))
            dc2 = dstate.get(c2, torch.zeros_like(state[c2]))

            dc1 -= flux * state[c2]
            dc2 += flux * state[c1]

            dstate[c1] = dc1
            dstate[c2] = dc2

    def forward(self, state: State[torch.Tensor]):
        dstate = State()
        self.diff(dstate, state)
        return dstate

    def _normal_meas_err(self, name: str, x: torch.Tensor):
        scale = self.noise_scale
        if x.ndim == 0:
            return pyro.sample(name, dist.Normal(x, scale))
        else:
            return pyro.sample(name, dist.Normal(x, scale).to_event(x.ndim))

    @pyro.nn.pyro_method
    def observation(self, state: State[torch.Tensor]):

        if np.isclose(self.noise_scale, 0.):
            return state

        for k, v in state.items():
            state[k] = self._normal_meas_err(f"{k}_obs", v)

        return state


def flux_prior(n_fips, n_states, n_pairs, mapped_state_idx):

    flux_plate = pyro.plate("flux_plate", n_pairs, dim=-3)
    states_plate = pyro.plate("states_plate", n_states, dim=-2)
    counties_plate = pyro.plate("counties_plate", n_fips, dim=-1)

    scale = 0.1

    with flux_plate:
        federal_flux = pyro.sample("federal_flux", dist.Normal(0., scale))

        with states_plate:
            state_flux = pyro.sample("state_flux", dist.Normal(federal_flux, scale))

        with counties_plate:
            # Could extend this with a mapped_obs_idx as well, and index into the last dimension.
            mapped_county_flux = state_flux[:, mapped_state_idx, ...]
            # Now, transpose because the county and state dimensions effectively get swapped by
            #  indexing into the state dimension.
            mapped_county_flux = mapped_county_flux.transpose(states_plate.dim, counties_plate.dim)
            county_flux = pyro.sample("county_flux", dist.Normal(mapped_county_flux, scale))

    # The fips codes are "flattened" wrt states, so just remove them, but note that the
    #  state flux sample still centers the county flux for counties in that state.
    # Returning counties plate as well so that initial_state sampling can use it. Kind of strange.
    return county_flux.squeeze(dim=states_plate.dim), counties_plate


def full_prior(employment_classes, fips_codes):
    mapped_state_codes = to_state_codes(fips_codes)
    unique_state_codes = list(set(mapped_state_codes))

    # Get a tensor that gives you the index of each mapped state code in the unique state codes.
    mapped_state_idx = tnsr([unique_state_codes.index(c) for c in mapped_state_codes])

    # Get each pair of employment categories.
    pairs = [pair for pair in product(employment_classes, employment_classes) if pair[0] != pair[1]]

    fluxes, counties_plate = flux_prior(
        n_fips=len(fips_codes),
        n_states=len(unique_state_codes),
        n_pairs=len(pairs),
        mapped_state_idx=mapped_state_idx
    )

    fdyns = FluxDynamics(fluxes, pairs, noise_scale=.05)

    with counties_plate:
        initial_state = State(**{
            ec: pyro.sample(f"{ec}0", dist.Uniform(0., .5)) for ec in employment_classes
        })

    return fdyns, initial_state


def run_svi_inference(model, n_steps=10, verbose=250, lr=0.03, **model_kwargs):
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % verbose == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide


def main():
    employment_classes = ["employed", "unemployed", "not_in_labor_force"]
    fips_codes = [1001, 1005, 2001, 2003, 3005]

    fdyns, initial_state = full_prior(employment_classes, fips_codes)

    start_time = tnsr(0.0)
    end_time = tnsr(30.0)
    step_size = tnsr(0.1)
    # noinspection PyTypeChecker
    trajectory_times = torch.arange(start_time + step_size, end_time, step_size)

    with LogTrajectory(trajectory_times) as traj:
        simulate(fdyns, initial_state, start_time, end_time, solver=TorchDiffEq())

    # Build a dataset by subsampling the noisy_traj for all states.
    noisy_traj = fdyns.observation(traj.trajectory)
    data = State(**{f"{k}_obs": v[:, ::10] for k, v in noisy_traj.items()})
    data_times = traj.times[::10]

    # plt.figure()
    # plt.plot(data_times, data["employed"][0])
    # plt.plot(data_times, data["unemployed"][1])
    # plt.plot(data_times, data["not_in_labor_force"][2])
    # plt.show()

    obs = condition(data=data)(fdyns.observation)

    def conditioned_model():
        fdyns, initial_state = full_prior(employment_classes, fips_codes)

        with StaticBatchObservation(times=data_times, observation=obs):
            simulate(fdyns, initial_state, start_time, end_time, solver=TorchDiffEq())

    with pyro.poutine.trace() as tr:
        conditioned_model()

    guide = run_svi_inference(conditioned_model, verbose=1, n_steps=10)

    return


if __name__ == "__main__":
    main()
