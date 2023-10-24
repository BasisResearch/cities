import pyro
import torch
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
import pandas as pd


class BayesSDID(pyro.nn.PyroModule):
    def __init__(self, X: pd.DataFrame):
        """
        Input:
            X: dataframe with columns "unit_index", "time_index", "in_treatment_group", "y", "treated"
        """
        super().__init__()
        # TODO: must be that in unit_index, the first 1, ... N_co indcs correspond to control units and
        # the remaining N_co + 1, ..., N correspond to treated units
        self.X = X.copy()
        units_by_group = self.X[["unit_index", "in_treatment_group"]].drop_duplicates()
        self.N = units_by_group.shape[0] # number of units
        self.N_tr = units_by_group["in_treatment_group"].sum()  # number of treated units
        self.N_co = self.N - self.N_tr  # number of control units
        self.T_pre = self.X[self.X["treated"] == 1]["time_index"].min()  # number of pre-treatment periods
        self.T_post = self.X["time_index"].max() - self.T_pre + 1  # number of post-treatment periods
        self.T = self.T_pre + self.T_post  # total number of periods
        self.times_by_units = torch.tensor(pd.pivot_table(self.X, values="y", index="time_index", columns="unit_index").values).float()
        self.avg_y_post_treat = self.times_by_units[self.T_pre:, :self.N_co].mean(axis=0)  # average of each control unit over the post-treatment period
        self.y_pre_treat_tr_avg = self.times_by_units[:self.T_pre, self.N_co:].mean(axis=1)
        self.y = torch.tensor(self.X["y"].values)
        self.treated = torch.tensor(self.X["treated"].values)
        self.unit_index = list(self.X["unit_index"].values)
        self.time_index = list(self.X["time_index"].values)
    
    def _get_module_param(self, param, module_ix):
        if len(param.shape) > 1:
            return param[module_ix].squeeze()
        return param

    def sample_synthetic_control_weights(self):
        w0 = pyro.sample("w0", dist.Normal(0, 1)) # intercept
        w_co = pyro.sample("w_co", dist.Dirichlet(torch.ones(self.N_co))) # convex combination of control units
        return w0, w_co
    
    def sample_time_weights(self):
        lam_0 = pyro.sample("lam_0", dist.Normal(0, 10)) # intercept
        lam_pre = pyro.sample("lam_pre", dist.Dirichlet(torch.ones(self.T_pre))) # convex combination of time periods
        return lam_0, lam_pre

    def sample_response_params(self, prior_scale=10):
        # Intercept, time fixed effects, treatment effect, unit fixed effects 
        mu = pyro.sample("mu", dist.Normal(0, prior_scale))
        beta = pyro.sample("beta", dist.Normal(0, prior_scale).expand((self.T,)).to_event(1))
        tau = pyro.sample("tau", dist.Normal(0, prior_scale))
        alpha = pyro.sample( "alpha", dist.Normal(0, prior_scale).expand((self.N,)).to_event(1))
        return mu, beta, tau, alpha
    
    def synthetic_control_unit(self, times_by_units: torch.Tensor, w0: torch.Tensor, w_co: torch.Tensor):
        return w0 + times_by_units.mv(w_co)
            
    def time_control(self, units_by_time: torch.Tensor, lam_0, lam_pre):
        return lam_0 + units_by_time.mv(lam_pre)

    def forward(self, **kwargs):        
        # Sample synthetic control weights, time weights, response parameters
        w0, w_co = self.sample_synthetic_control_weights()
        _shape_w_tr = list(w_co.shape)
        _shape_w_tr[-1] = self.N_tr
        w_co_tr = torch.cat([w_co, 1 / self.N_tr * torch.ones(_shape_w_tr)], axis=-1) # TODO: this assumes
        lam_0, lam_pre = self.sample_time_weights()
        _shape_lam_post = list(w_co.shape)
        _shape_lam_post[-1] = self.T_post
        lam_pre_post = torch.cat([lam_pre, 1 / self.T_post * torch.ones(_shape_lam_post)], axis=-1) # TODO: this assumes
        mu, beta, tau, alpha = self.sample_response_params()
                
        y_sc = self.synthetic_control_unit(
            self.times_by_units[:self.T_pre, :self.N_co], 
            self._get_module_param(w0, 0),
            self._get_module_param(w_co, 0)
        )
        
        with pyro.plate("synthetic_control_weights", self.T_pre):
            pyro.sample("y_pre_treat_tr_avg", dist.Normal(y_sc, 1.0), obs=self.y_pre_treat_tr_avg)

        # Time weights likelihood
        y_time = self.time_control(
            self.times_by_units[:self.T_pre, :].T, 
            self._get_module_param(lam_0, 0),
            self._get_module_param(lam_pre, 0)
        )

        with pyro.plate("time_weights", self.N_co):
            pyro.sample("avg_y_post_treat", dist.Normal(y_time[:self.N_co], 1.0), obs=self.avg_y_post_treat)

        # Response likelihood
        # Here we use the copy of module one parameters to response likelihood to change module one 
        # gradients
        weights = self._get_module_param(w_co_tr, 1)[self.unit_index] * self._get_module_param(lam_pre_post, 1)[self.time_index]
        f = self._get_module_param(mu, 1) + self._get_module_param(beta, 1)[self.time_index] + self._get_module_param(alpha, 1)[self.unit_index] + self._get_module_param(tau, 1) * self.treated
        with pyro.plate("response", self.N * self.T):
            pyro.sample("y", dist.Normal(f, 1 / weights), obs=self.y)





# Define a helper function to run SVI.
def run_svi_inference(model, n_steps=100, verbose=True, lr=.03, vi_family=AutoNormal, guide=None, **model_kwargs):
    if guide is None:
        guide = vi_family(model)
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
        if (step % 1000 == 0) or (step == 1) & verbose:
            print("[iteration %04d] loss: %.4f" % (step, loss))
    return guide