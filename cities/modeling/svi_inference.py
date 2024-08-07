import matplotlib.pyplot as plt
import pyro
import torch
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean


def run_svi_inference(
    model,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    guide=None,
    hide=[],
    n_steps=500,
    ylim=None,
    plot = True,
    **model_kwargs
):
    losses = []
    if guide is None:
        guide = vi_family(
            pyro.poutine.block(model, hide=hide), init_loc_fn=init_to_mean
        )
    elbo = pyro.infer.Trace_ELBO()(model, guide)

    elbo(**model_kwargs)
    adam = torch.optim.Adam(elbo.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        adam.zero_grad()
        loss = elbo(**model_kwargs)
        loss.backward()
        losses.append(loss.item())
        adam.step()
        if (step % 50 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))

    if plot:
        plt.plot(losses)
        if ylim:
            plt.ylim(ylim)
        plt.show()

    return guide
