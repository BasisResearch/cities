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
    def __init__(self, flux_rates: Tnsr, noise_scale: float):
        super().__init__()
        self.flux_rates = flux_rates  # .shape == (n_classes, n_classes, n_fips)
        self.noise_scale = noise_scale

    @pyro.nn.pyro_method
    def diff(self, dstate: State[torch.Tensor], state: State[torch.Tensor]) -> None:

        # Flux for a particular pair gets multiplied by the source state (indexed by row).
        totflux = state["fluxers"][:, None, :] * self.flux_rates  # .shape == (n_classes, n_classes, n_fips)

        # Summing over the columns (for row totals) gives the total going into the state indexed by the row.
        fluxin = totflux.sum(dim=0)  # .shape == (n_fips, n_classes)

        # Summing over the rows (for column totals) gives the total going out of the state indexed by the column.
        fluxout = totflux.sum(dim=1)  # .shape == (n_fips, n_classes)

        dstate["fluxers"] = fluxin - fluxout

        inoutsum = dstate["fluxers"].sum(dim=0)
        assert torch.allclose(inoutsum, torch.zeros_like(inoutsum), atol=1e-6), f"Fluxers not conserved: {inoutsum}"

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


class TotalLaborForceDynamics(FluxDynamics):

    def __init__(self, *args, total_labor_force_f_of_t, lf_fluxes, **kwargs):
        # The tlf function takes in time as a scalar and returns a state vector representing
        #  the total labor force across all fips.
        self.tlf = total_labor_force_f_of_t
        self.lf_fluxes = lf_fluxes  # .shape == (n_cats, n_fips)

        super().__init__(*args, **kwargs)

    @pyro.nn.pyro_method
    def diff(self, dstate: State[torch.Tensor], state: State[torch.Tensor]) -> None:
        super().diff(dstate, state)
        t = state['t']
        dpopdt = torch.autograd.grad(self.tlf(t), t)[0]  # .shape == (n_fips,)

        # For a given county, the labor force deltas are distributed across the employment categories,
        #  so these need to be normalized across categories.
        normed_lf_fluxes = self.lf_fluxes / self.lf_fluxes.sum(dim=0, keepdim=True)

        # Note that this vectorization is over (n_fips,), so flux here won't be normalized.
        for flux, k in zip(normed_lf_fluxes, state.keys()):
            dstate[k] += flux * dpopdt


def transform_county_fluxes(county_flux, end_scale):
    return torch.sigmoid(county_flux) * end_scale


def flux_prior(n_fips, n_states, n_pairs, mapped_state_idx):

    flux_plate = pyro.plate("flux_plate", n_pairs, dim=-3)
    states_plate = pyro.plate("states_plate", n_states, dim=-2)
    counties_plate = pyro.plate("counties_plate", n_fips, dim=-1)

    # This is designed to have three chained normals not grossly violate the domain
    #  of a sigmoid.
    scale = 0.5

    # After pushing through a sigmoid, this scales things from unit range.
    end_scale = 0.1

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
    county_flux = county_flux.squeeze(dim=states_plate.dim)

    return transform_county_fluxes(county_flux, end_scale), counties_plate


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

    with counties_plate:
        initial_state = State(**{
            ec: pyro.sample(f"{ec}0", dist.Uniform(0.5, 1.0)) for ec in employment_classes
        })

        # pop_growth = pyro.sample("pop_growth", dist.Normal(0.0, 0.1))
        # pop_start =

    # fdyns = TotalLaborForceDynamics(
    #     fluxes,
    #     pairs,
    #     total_labor_force_f_of_t=,
    #     lf_fluxes=,
    #     noise_scale = .05
    # )
    fdyns = FluxDynamics(
        fluxes,
        pairs,
        noise_scale=0.05
    )

    return fdyns, initial_state


def run_svi_inference(model, n_steps=10, verbose=250, lr=0.03, **model_kwargs):
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    elbo = pyro.infer.Trace_ELBO()(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    losses = []
    # Do gradient steps
    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        losses.append(loss)
        loss.backward()
        adam.step()
        if (step % verbose == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide, losses


def main():
    employment_classes = ["employed", "unemployed", "not_in_labor_force"]
    fips_codes = [1001, 1005, 2001, 2003, 3005]

    fdyns, initial_state = full_prior(employment_classes, fips_codes)

    start_time = tnsr(0.0)
    end_time = tnsr(30.0)
    step_size = tnsr(0.01)
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


def dynsys_only():
    employment_classes = ["agriculture", "knowledge", "healthcare"]
    _colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    colors = dict(zip(employment_classes, _colors))
    # fips_codes = [1001, 1005, 2001, 2003, 3005]
    fips_codes = [1001, 1002]

    flux_rates = torch.rand(
        len(employment_classes),
        len(employment_classes),
        len(fips_codes)
        # With diagonal zeroed, as self flux terms aren't necessary.
    ) * (1. - torch.eye(len(employment_classes)))[:, :, None]

    assert (flux_rates >= 0).all()

    fdyns = FluxDynamics(
        flux_rates=flux_rates,
        noise_scale=0.01
    )

    initial_state = State(
        fluxers=torch.rand(len(employment_classes), len(fips_codes)),
    )

    start_time = tnsr(0.0)
    end_time = tnsr(30.0)
    step_size = tnsr(0.1)
    # noinspection PyTypeChecker
    trajectory_times = torch.arange(start_time + step_size, end_time, step_size)

    with LogTrajectory(trajectory_times) as traj:
        simulate(fdyns, initial_state, start_time, end_time, solver=TorchDiffEq(), method="dopri8")

    fc = 0
    for i in range(len(employment_classes)):
        plt.plot(traj.times, traj.trajectory["fluxers"][i, fc], color=colors[employment_classes[i]])

    plt.show()


if __name__ == "__main__":
    # main()
    dynsys_only()
