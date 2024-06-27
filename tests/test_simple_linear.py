import torch
from chirho.interventional.handlers import do
from chirho.robust.handlers.predictive import PredictiveModel

import pyro
from cities.modeling.simple_linear import SimpleLinear, SimpleLinearRegisteredInput
from cities.modeling.svi_inference import run_svi_inference
from pyro.infer import Predictive

from chirho.counterfactual.handlers import MultiWorldCounterfactual



# set up the data
n = 600
part = n // 3
n_steps = 600

x_cat = torch.cat(
    [
        torch.zeros([part], dtype=torch.long),
        torch.ones([part], dtype=torch.long),
        2 * torch.ones([part], dtype=torch.long),
    ]
)
x_con = torch.randn(n)

y_cat = x_cat * 2 + torch.randn(n) * 0.05
y_con = x_con * 2 + torch.randn(n) * 0.05
y_mixed = x_cat * 2 + x_con * 2 + torch.randn(n) * 0.05

x_cat_test = torch.tensor([0, 2, 0, 1])
x_con_test = torch.tensor([-1.0, -1.0, 1.0, 1.0])
y_test = x_cat_test * 2 + x_con_test * 2

# expected mean predictions on test data
target = torch.tensor([-2.0, 2.0, 2.0, 4.0])

model_kwargs_cat = {"categorical": {"x_cat": x_cat}, "continuous": {}, "outcome": y_cat}
model_kwargs_con2 = {"categorical": {}, "continuous": {"x_con": x_con, "x_con2": x_con}, "outcome": y_con}
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

    guide_cat = run_svi_inference(
        simple_linear_cat, n_steps=n_steps, lr=0.01, verbose=True, **model_kwargs_cat
    )

    predictive_cat = Predictive(
        simple_linear_cat, guide=guide_cat, num_samples=1000, parallel=True
    )
    samples_cat = predictive_cat(**model_kwargs_cat)
    cat_0 = samples_cat["weights_categorical_x_cat"].squeeze()[:,0]
    cat_1 = samples_cat["weights_categorical_x_cat"].squeeze()[:,1] 
    assert cat_1.mean().item() - cat_0.mean().item() > 1


def test_simple_linear_con():

    pyro.clear_param_store()

    print(model_kwargs_con2['continuous'].keys())
    simple_linear_con = SimpleLinear(
        **model_kwargs_con2
    )

    #    simple_linear_con = condition(simple_linear_con, data =  {"outcome_observed": y_con})

    guide_con = run_svi_inference(
        simple_linear_con, n_steps=n_steps, lr=0.01, verbose=True, **model_kwargs_con2
    )

    predictive_con = Predictive(
        simple_linear_con, guide=guide_con, num_samples=1000, parallel=True
    )
    samples_con = predictive_con(**model_kwargs_con2)
    con = samples_con["weight_continuous"].squeeze()
    assert con.mean() > -3
    assert con.mean() < 3


#will re-use these in later tests
simple_linear_mixed2 = SimpleLinear(
    categorical={"x_cat": x_cat}, continuous={"x_con": x_con}, outcome=y_mixed
)

pyro.clear_param_store()

guide_mixed2 = run_svi_inference(
    simple_linear_mixed2, n_steps=n_steps, lr=0.01, verbose=True, **model_kwargs_mixed
)


def test_simple_linear_mixed():

    predictive_mixed = Predictive(
        simple_linear_mixed2,
        guide=guide_mixed2,
        num_samples=1000,
    )  # parallel=True) # fails for extraneous data

    samples_mixed = predictive_mixed(**model_kwargs_mixed)

#    mixed_cat_0 = samples_mixed["weights_categorical_x_cat"][:, :, 0].squeeze()
 #   mixed_cat_1 = samples_mixed["weights_categorical_x_cat"][:, :, 1].squeeze()

    mixed_cat_0 = samples_mixed["weights_categorical_x_cat"].squeeze()[:,0]
    mixed_cat_1 = samples_mixed["weights_categorical_x_cat"].squeeze()[:,1] 

    assert mixed_cat_1.mean() - mixed_cat_0.mean() > 1.2
    assert samples_mixed["weight_continuous"].squeeze().mean() > 1.2

    samples_test = predictive_mixed(
        {"x_cat": x_cat_test}, continuous={"x_con": x_con_test}, outcome=None
    )
    outcome_preds = samples_test["outcome_observed"].squeeze().mean(axis=0)
    assert torch.allclose(outcome_preds, target, atol=0.5)


#################################################
# register inputs with PredictiveModel, intervene
#################################################


def test_SimpleLinearRegisteredInput():

    predictive_model = PredictiveModel(simple_linear_mixed2, guide=guide_mixed2)

    # predictive_model_registered = SimpleLinearRegisteredInput(
    #     predictive_model,
    #     categorical={"x_cat": x_cat_test},
    #     continuous={"x_con": x_con_test},
    # )

    # print(x_cat_test.shape)

    #with MultiWorldCounterfactual() as mwc:
    # with pyro.poutine.trace() as tr:
    #     with do(actions = {'categorical_x_cat': torch.tensor([1]).expand(x_cat_test.shape[-1])}):
    #         before = predictive_model_registered()

    #print(tr.trace.nodes.keys())

    with pyro.poutine.trace():
        before = predictive_model()

    assert torch.allclose(before, target, atol=2.5)

    int = torch.tensor([1])

    int_exp = int.expand(torch.tensor(4))

    print(int_exp)

    with do(
        actions={
            "categorical_x_cat": torch.tensor([1, 1, 1, 1]),
            "continuous_x_con": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
    ):
        with pyro.poutine.trace():
            after = predictive_model_registered()

    with MultiWorldCounterfactual(first_available_dim= -10) as mwc:
        with do( actions={
                "categorical_x_cat": int_exp,
                "continuous_x_con": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            }
        ):
                with pyro.poutine.trace():
                    after = predictive_model_registered()

    # target_after = torch.tensor([4.0, 4.0, 4.0, 4.0])

    # assert torch.allclose(after, target_after, atol=3)

    #this will fail!
    # with MultiWorldCounterfactual(first_available_dim=-5) as mwc:
    #     with do(actions = {'categorical_x_cat': torch.tensor(1)}):
    #         with pyro.poutine.trace() as tr:
    #             predictive_model_registered()


test_SimpleLinearRegisteredInput()