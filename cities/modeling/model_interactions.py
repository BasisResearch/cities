import pyro
import pyro.distributions as dist
import torch


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
