import torch

import pyro
from cities.modeling.simple_linear import SimpleLinear
from cities.modeling.svi_inference import run_svi_inference
from pyro.infer import Predictive
from chirho.observational.handlers.condition import condition

# set up the data
n = 600
half = n // 2
n_steps = 600

x_cat = torch.cat(
    [torch.zeros([half], dtype=torch.long), torch.ones([half], dtype=torch.long)]
)
x_con = torch.randn(n)

y_cat = x_cat * 2 + torch.randn(n) * 0.05
y_con = x_con * 2 + torch.randn(n) * 0.05
y_mixed = x_cat * 2 + x_con * 2 + torch.randn(n) * 0.05

x_cat_test = torch.tensor([0, 1, 0, 1])
x_con_test = torch.tensor([-1., -1., 1., 1. ])
y_test = x_cat_test * 2 + x_con_test * 2



model_kwargs_cat = {"categorical": {"x_cat": x_cat}, "continuous": {}, "outcome": y_cat}
model_kwargs_con = {"categorical": {}, "continuous": {"x_con": x_con}, "outcome": y_con}
model_kwargs_mixed = {
    "categorical": {"x_cat": x_cat},
    "continuous": {"x_con": x_con},
    "outcome": y_mixed,
}


def test_simple_linear_cat():

    pyro.clear_param_store()

    simple_linear_cat = SimpleLinear(
        categorical={"x_cat": x_cat}, continuous={}, outcome=y_cat
    )

# ideally, once we can ChiRho uncondition
#    simple_linear_cat = condition(simple_linear_cat, data =  {"outcome_observed": y_cat})


    guide_cat = run_svi_inference(
        simple_linear_cat, n_steps=n_steps, lr=0.01, verbose=True, **model_kwargs_cat
    )

    predictive_cat = Predictive(
        simple_linear_cat, guide=guide_cat, num_samples=1000, parallel=True
    )
    samples_cat = predictive_cat(**model_kwargs_cat)
    cat_0 = samples_cat["weights_categorical_x_cat"][:, 0].squeeze()
    cat_1 = samples_cat["weights_categorical_x_cat"][:, 1].squeeze()
    assert cat_1.mean() - cat_0.mean() > 1.2


def test_simple_linear_con():

    pyro.clear_param_store()

    simple_linear_con = SimpleLinear(
        categorical={}, continuous={"x_con": x_con}, outcome=y_con
    )


#    simple_linear_con = condition(simple_linear_con, data =  {"outcome_observed": y_con})

    guide_con = run_svi_inference(
        simple_linear_con, n_steps=n_steps, lr=0.01, verbose=True, **model_kwargs_con
    )

    predictive_con = Predictive(
        simple_linear_con, guide=guide_con, num_samples=1000, parallel=True
    )
    samples_con = predictive_con(**model_kwargs_con)
    con = samples_con["weight_continuous"].squeeze()
    assert con.mean() > 1.2


def test_simple_linear_mixed():

    pyro.clear_param_store()

    simple_linear_mixed = SimpleLinear(
        categorical={"x_cat": x_cat}, continuous={"x_con": x_con}, outcome=y_mixed
    )

#    simple_linear_mixed = condition(simple_linear_mixed, data =  {"outcome_observed": y_mixed})

    guide_mixed = run_svi_inference(
        simple_linear_mixed,
        n_steps=n_steps,
        lr=0.01,
        verbose=True,
        **model_kwargs_mixed
    )

    predictive_mixed = Predictive(
        simple_linear_mixed, guide=guide_mixed, num_samples=1000,) # parallel=True) # fails for extraneous data
    
    samples_mixed = predictive_mixed(**model_kwargs_mixed)

    mixed_cat_0 = samples_mixed["weights_categorical_x_cat"][:, :, 0].squeeze()
    mixed_cat_1 = samples_mixed["weights_categorical_x_cat"][:, :, 1].squeeze()

    assert mixed_cat_1.mean() - mixed_cat_0.mean() > 1.2
    assert samples_mixed["weight_continuous"].squeeze().mean() > 1.2

    samples_test = predictive_mixed({'x_cat': x_cat_test}, continuous = {'x_con': x_con_test}, outcome = None)
    outcome_preds = samples_test['outcome_observed'].squeeze().mean(axis=0)
    target = torch.tensor([-2.,0.,2.,4.])
    assert torch.allclose(outcome_preds, target, atol=0.1)


