import matplotlib.pyplot as plt
import torch

import pyro
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean


def run_svi_inference(
    model,
    n_steps=500,
    verbose=True,
    lr=0.03,
    vi_family=AutoMultivariateNormal,
    guide=None,
    **model_kwargs
):
    losses = []
    if guide is None:
        guide = vi_family(model, init_loc_fn=init_to_mean)
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

    plt.plot(losses)

    return guide
