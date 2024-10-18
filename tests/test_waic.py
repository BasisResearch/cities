import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide import AutoMultivariateNormal

from cities.modeling.waic import compute_waic

num_steps = 400
max_plate_nesting = 9


torch.manual_seed(0)
X = torch.linspace(0, 1, 10000)
y = 3 * X + torch.randn(10000) * 0.1


# TODO refactor to use with the repo general svi inference
# TODO rename models to something more meaningful

# Key idea: we have two models, one with a linear relationship and
# one with a quadratic relationship
# We generate synthetic linear data, train the models
# waic should be much better for the linear one


class Submodel1(pyro.nn.PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, X, y=None):
        n = X.shape[0]

        slope_1 = pyro.sample("slope_1", dist.Normal(0.0, 2.0))
        intercept_1 = pyro.sample("intercept_1", dist.Normal(0.0, 1.0))
        sigma_1 = pyro.sample("sigma_1", dist.HalfCauchy(1.0))

        mean_1 = slope_1 * X + intercept_1

        with pyro.plate("data_1", n):
            return pyro.sample("obs_1", dist.Normal(mean_1, sigma_1), obs=y)


submodel1 = Submodel1()


class Submodel2(pyro.nn.PyroModule):
    def __init__(self):
        super().__init__()

    def forward(self, X, y=None):
        n = X.shape[0]

        intercept_2 = pyro.sample("intercept_2", dist.Normal(0.0, 1.0))
        quadratic_2 = pyro.sample("quadratic_2", dist.Normal(0.0, 1.0))
        sigma_2 = pyro.sample("sigma_2", dist.HalfCauchy(1.0))

        mean_2 = quadratic_2 * X**2 + intercept_2

        with pyro.plate("data_2", size=n):
            return pyro.sample("obs_2", dist.Normal(mean_2, sigma_2), obs=y)


submodel2 = Submodel2()


def run_svi_inference(
    model,
    num_steps=num_steps,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    elbo=pyro.infer.Trace_ELBO(),
    discrete_sites=[],
    guide=None,
    plot_loss=True,
    obs_n=1,
    **model_kwargs
):
    if guide is None:
        guide = vi_family(pyro.poutine.block(model, hide=discrete_sites))
    elbo = elbo(model, guide)
    # initialize parameters
    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)
    # Do gradient steps
    losses = []
    for step in range(1, num_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        adam.step()
        if (step % 100 == 0) or (step == 1) & verbose:
            print(
                "[iteration %04d] loss: %.4f" % (step, loss),
                "avg loss: ",
                round(loss.item() / obs_n),
            )
        losses.append(loss.item() / obs_n)
    if plot_loss:
        plt.plot(losses)
    return guide, losses


pyro.clear_param_store()

guide_1, losses_1 = run_svi_inference(submodel1, X=X, y=y, num_steps=num_steps)

waic_linear = compute_waic(
    model=submodel1,
    guide=guide_1,
    num_particles=1000,
    sites=["obs_1"],
    max_plate_nesting=max_plate_nesting,
    X=X,
    y=y,
)["waic"]

pyro.clear_param_store()
guide_2, losses_2 = run_svi_inference(submodel2, X=X, y=y, num_steps=num_steps)

waic_quadratic = compute_waic(
    model=submodel2,
    guide=guide_2,
    num_particles=1000,
    sites=["obs_2"],
    max_plate_nesting=max_plate_nesting,
    X=X,
    y=y,
)["waic"]


def test_waic():

    assert waic_linear < 4 * waic_quadratic
