import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyro
import torch

from cities.modeling.model_interactions import model_cities_interaction
from cities.modeling.modeling_utils import (
    prep_wide_data_for_inference,
    revert_prediction_df,
    revert_standardize_and_scale_scaler,
)
from cities.utils.cleaning_utils import find_repo_root
from cities.utils.data_grabber import DataGrabber


class CausalInsight:
    def __init__(
        self,
        outcome_dataset,
        intervention_dataset,
        num_samples=1000,
        sites=None,
        smoke_test=None,
    ):
        self.outcome_dataset = outcome_dataset
        self.intervention_dataset = intervention_dataset
        self.root = find_repo_root()
        self.num_samples = num_samples
        self.data = None
        self.smoke_test = smoke_test

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # if sites is None:
        #     self.sites = ["weight_TY"]
        # else:
        #    self.sites = sites

        self.tau_samples_path = os.path.join(
            self.root,
            "data/tau_samples",
            f"{self.intervention_dataset}_{self.outcome_dataset}_{self.num_samples}_tau.pkl",
        )

    def load_guide(self, forward_shift):
        pyro.clear_param_store()
        guide_name = (
            f"{self.intervention_dataset}_{self.outcome_dataset}_{forward_shift}"
        )
        guide_path = os.path.join(
            self.root, "data/model_guides", f"{guide_name}_guide.pkl"
        )

        with open(guide_path, "rb") as file:
            self.guide = dill.load(file)
        param_path = os.path.join(
            self.root, "data/model_guides", f"{guide_name}_params.pth"
        )

        pyro.get_param_store().load(param_path)

        self.forward_shift = forward_shift

    def generate_samples(self):
        self.data = prep_wide_data_for_inference(
            outcome_dataset=self.outcome_dataset,
            intervention_dataset=self.intervention_dataset,
            forward_shift=self.forward_shift,
        )
        self.model = model_cities_interaction

        self.model_args = self.data["model_args"]

        self.model_conditioned = pyro.condition(
            self.model,
            data={"T": self.data["t"], "Y": self.data["y"], "X": self.data["x"]},
        )

        self.predictive = pyro.infer.Predictive(
            model=self.model,
            guide=self.guide,
            num_samples=self.num_samples,
            parallel=True,
            # return_sites=self.sites,
        )
        self.samples = self.predictive(*self.model_args)

        # idexing and gathering with mwc in this context
        # seems to fail, calculating the expected diff made by the intervention manually
        # wrt to actual observed outcomes rather than predicting outcomes themselves
        # effectively keeping the noise fixed and focusing on a counterfactual claim

        # TODO possible delete in the current strategy deemed uncontroversial
        # else:
        #     if not isinstance(intervened_value, torch.Tensor):
        #         intervened_value = torch.tensor(intervened_value, device=self.device)
        #     intervened_expanded = intervened_value.expand_as(self.data['t'])

        #     with MultiWorldCounterfactual(first_available_dim=-6) as mwc:
        #         with do(actions = dict(T = intervened_expanded)):
        #                 self.predictive = pyro.infer.Predictive(model=self.model, guide=self.guide,
        #                         num_samples=self.num_samples, parallel=True)
        #                 self.samples = self.predictive(*self.model_args)
        #     self.mwc = mwc

    def generate_tensed_samples(self):
        self.tensed_samples = {}
        self.tensed_tau_samples = {}

        for shift in [1, 2, 3]:
            self.load_guide(shift)
            self.generate_samples()
            self.tensed_samples[shift] = self.samples
            self.tensed_tau_samples[shift] = (
                self.samples["weight_TY"].squeeze().detach().numpy()
            )

        if not self.smoke_test:
            if not os.path.exists(self.tau_samples_path):
                with open(self.tau_samples_path, "wb") as file:
                    dill.dump(self.tensed_tau_samples, file)

    def get_tau_samples(self):
        if os.path.exists(self.tau_samples_path):
            with open(self.tau_samples_path, "rb") as file:
                self.tensed_tau_samples = dill.load(file)
        else:
            raise ValueError("No tau samples found. Run generate_tensed_samples first.")

    def get_fips_predictions(self, fips, intervened_value, year=None):
        self.fips = fips
        self.intervened_value = intervened_value
        if self.data is None:
            self.data = prep_wide_data_for_inference(
                outcome_dataset=self.outcome_dataset,
                intervention_dataset=self.intervention_dataset,
                forward_shift=3,  # shift doesn't matter here, as long as data exists
            )

        # start with the latest year possible by default
        if year is None:
            year = self.data["years_available"][-1]
        assert year in self.data["years_available"]

        self.year = year

        # find years for prediction
        outcome_years = self.data["outcome_years"]
        year_id = [int(x) for x in outcome_years].index(year)
        self.year_id = year_id

        self.prediction_years = outcome_years[(year_id) : (year_id + 4)]

        # find fips unit index
        dg = DataGrabber()
        dg.get_features_std_wide([self.intervention_dataset, self.outcome_dataset])
        dg.get_features_wide([self.intervention_dataset])
        interventions_this_year_original = dg.wide[self.intervention_dataset][str(year)]

        self.intervened_value_original = revert_standardize_and_scale_scaler(
            self.intervened_value, self.year, self.intervention_dataset
        )

        # interventions_this_year = dg.std_wide[self.intervention_dataset][str(year)]

        # nearest_intervention_index = min(
        #     range(len(interventions_this_year)),
        #     key=lambda i: abs(interventions_this_year[i] - self.intervened_value),
        # )
        # self.intervened_value_original = interventions_this_year_original[
        #     nearest_intervention_index
        # ]

        self.intervened_value_percentile = round(
            (
                np.mean(
                    interventions_this_year_original.values
                    <= self.intervened_value_original
                )
                * 100
            ),
            1,
        )

        self.fips_id = (
            dg.std_wide[self.intervention_dataset]
            .loc[dg.std_wide[self.intervention_dataset]["GeoFIPS"] == fips]
            .index[0]
        )

        self.name = dg.std_wide[self.intervention_dataset]["GeoName"].iloc[self.fips_id]

        # get observed values at the prediction times
        self.observed_intervention = dg.std_wide[self.intervention_dataset].iloc[
            self.fips_id
        ][str(year)]

        self.observed_intervention_original = dg.wide[self.intervention_dataset].iloc[
            self.fips_id
        ][str(year)]

        self.observed_outcomes = dg.std_wide[self.outcome_dataset].iloc[self.fips_id][
            outcome_years[year_id : (year_id + 4)]
        ]
        self.intervention_diff = self.intervened_value - self.observed_intervention

        self.intervention_impact = {}
        self.intervention_impact_mean = []
        self.intervention_impact_low = []
        self.intervention_impact_high = []
        for shift in [1, 2, 3]:
            self.intervention_impact[shift] = (
                self.tensed_tau_samples[shift] * self.intervention_diff
            )
            self.intervention_impact_mean.append(
                np.mean(self.intervention_impact[shift])
            )
            self.intervention_impact_low.append(
                np.percentile(self.intervention_impact[shift], 2.5)
            )
            self.intervention_impact_high.append(
                np.percentile(self.intervention_impact[shift], 97.5)
            )

        predicted_mean = [self.observed_outcomes.iloc[0]] + (
            self.intervention_impact_mean + self.observed_outcomes.iloc[1:]
        ).tolist()
        predicted_low = [self.observed_outcomes.iloc[0]] + (
            self.intervention_impact_low + self.observed_outcomes.iloc[1:]
        ).tolist()
        predicted_high = [self.observed_outcomes.iloc[0]] + (
            self.intervention_impact_high + self.observed_outcomes.iloc[1:]
        ).tolist()

        self.predictions = pd.DataFrame(
            {
                "year": self.prediction_years,
                "observed": self.observed_outcomes,
                "mean": predicted_mean,
                "low": predicted_low,
                "high": predicted_high,
            }
        )

        self.predictions_original = revert_prediction_df(
            self.predictions, self.outcome_dataset
        )

        # TODO for some reason indexing using gather doesn't pick the right indices
        # look into this some time, do this by hand for now
        # with self.mwc:
        #     self.tau_samples = self.samples['weight_TY'].squeeze().detach().numpy()
        #     self.tensed_observed_samples[shift] = self.tensed_intervened_samples[shift] = gather(
        #     self.samples['Y'], IndexSet(**{"T": {0}}),
        #     event_dim=0,).squeeze()
        #     self.tensed_intervened_samples[shift] = gather(
        #     self.samples['Y'], IndexSet(**{"T": {1}}),
        #     event_dim=0,).squeeze()#[:,self.fips_id]

        #     self.tensed_outcome_difference[shift] = (
        #     self.tensed_intervened_samples[shift] - self.tensed_observed_samples[shift]
        #     )
        return

    def plot_predictions(
        self, range_multiplier=1.5, show_figure=True, scaling="transformed"
    ):
        assert scaling in ["transformed", "original"]

        dg = DataGrabber()

        if scaling == "transformed":
            dg.get_features_std_long([self.outcome_dataset])
            plot_data = dg.std_long[self.outcome_dataset]
            self.fips_observed_data = plot_data[
                plot_data["GeoFIPS"] == self.fips
            ].copy()

            y_min = (
                min(
                    self.fips_observed_data["Value"].min(),
                    self.predictions["low"].min(),
                )
                - 0.05
            )
            y_max = (
                max(
                    self.fips_observed_data["Value"].max(),
                    self.predictions["high"].max(),
                )
                + 0.05
            )
        else:
            dg.get_features_long([self.outcome_dataset])
            plot_data = dg.long[self.outcome_dataset]

            self.fips_observed_data = plot_data[
                plot_data["GeoFIPS"] == self.fips
            ].copy()

            y_min = 0.8 * min(
                self.fips_observed_data["Value"].min(),
                self.predictions_original["low"].min(),
            )
            y_max = 1.3 * max(
                self.fips_observed_data["Value"].max(),
                self.predictions_original["high"].max(),
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.fips_observed_data["Year"],
                y=self.fips_observed_data["Value"],
                mode="lines+markers",
                name=self.fips_observed_data["GeoName"].iloc[0],
                line=dict(color="darkred", width=3),
                text=self.fips_observed_data["GeoName"].iloc[0],
                textposition="top right",
                showlegend=False,
            )
        )

        if scaling == "transformed":
            fig.add_trace(
                go.Scatter(
                    x=self.predictions["year"],
                    y=self.predictions["mean"],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="mean prediction",
                    text=self.predictions["mean"],
                )
            )

            credible_interval_trace = go.Scatter(
                x=pd.concat([self.predictions["year"], self.predictions["year"][::-1]]),
                y=pd.concat([self.predictions["high"], self.predictions["low"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% credible interval around mean",
            )

        else:
            fig.add_trace(
                go.Scatter(
                    x=self.predictions_original["year"],
                    y=self.predictions_original["mean"],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="mean prediction",
                    text=self.predictions_original["mean"],
                )
            )

            credible_interval_trace = go.Scatter(
                x=pd.concat(
                    [
                        self.predictions_original["year"],
                        self.predictions_original["year"][::-1],
                    ]
                ),
                y=pd.concat(
                    [
                        self.predictions_original["high"],
                        self.predictions_original["low"][::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(255, 255, 255, 0.31)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% credible interval around mean",
            )

        fig.add_trace(credible_interval_trace)

        if scaling == "transformed":
            title = (
                f"Predicted {self.outcome_dataset} in {self.name} under intervention {self.intervened_value} "
                f"in year {self.year}<br>"
                f"compared to the observed values under observed intervention "
                f"{round(self.observed_intervention, 3)}."
            )

        else:
            title = (
                f"Predicted {self.outcome_dataset} in {self.name}<br>"
                f"under intervention {self.intervened_value_original}"
                f" in year {self.year}<br>"
                f"{self.intervened_value_percentile}% of counties received a lower intervention <br>"
                f"observed intervention: {self.observed_intervention_original}"
            )

        fig.update_yaxes(range=[y_min, y_max])

        fig.update_layout(
            title=title,
            title_font=dict(size=12),
            xaxis_title="Year",
            yaxis_title="Value",
            template="simple_white",
            legend=dict(x=0.05, y=1, traceorder="normal", orientation="h"),
        )

        self.predictions_plot = fig

        if show_figure:
            fig.show()
        else:
            return fig

    def plot_residuals(self):
        predictions = self.samples["Y"].squeeze()
        self.average_predictions = torch.mean(predictions, dim=0)
        plt.hist(self.average_predictions - self.data["y"].squeeze(), bins=70)
        plt.show()

    def predictive_check(self):
        y_flat = self.data["y"].view(-1)
        observed_mean = torch.mean(y_flat)
        tss = torch.sum((y_flat - observed_mean) ** 2)
        average_predictions_flat = self.average_predictions.view(-1)
        rss = torch.sum((y_flat - average_predictions_flat) ** 2)
        r_squared = 1 - (rss / tss)
        plt.scatter(average_predictions_flat, y_flat)
        plt.title(
            f"{self.intervention_dataset}, {self.outcome_dataset}." f"R2={r_squared}"
        )
        # TODO round r2
        plt.show

    def estimate_ATE(self):
        tau_samples = self.samples["weight_TY"].squeeze().detach().numpy()
        plt.hist(tau_samples, bins=70)
        plt.axvline(
            x=tau_samples.mean(),
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"mean = {tau_samples.mean():.3f}",
        )
        plt.title(
            f"ATE for {self.intervention_dataset}  and  {self.outcome_dataset} with forward shift = {self.forward_shift}"
        )
        plt.legend()
        plt.show()
