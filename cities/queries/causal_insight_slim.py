import os

import dill
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

from cities.utils.cleaning_utils import (
    revert_prediction_df,
    revert_standardize_and_scale_scaler,
    sigmoid,
)
from cities.utils.data_grabber import DataGrabber, find_repo_root
from cities.utils.percentiles import transformed_intervention_from_percentile


class CausalInsightSlim:
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
        self.smoke_test = smoke_test
        self.data = None

        self.tau_samples_path = os.path.join(
            self.root,
            "data/tau_samples",
            f"{self.intervention_dataset}_{self.outcome_dataset}_{self.num_samples}_tau.pkl",
        )

    def get_tau_samples(self):
        if os.path.exists(self.tau_samples_path):
            with open(self.tau_samples_path, "rb") as file:
                self.tensed_tau_samples = dill.load(file)
        else:
            raise ValueError("No tau samples found. Run generate_tensed_samples first.")

    def slider_values_to_interventions(self, intervened_percent, year):
        try:
            original_column = dg.wide[self.intervention_dataset][
                str(year)
            ].values.reshape(-1, 1)
        except NameError:
            dg = DataGrabber()
            dg.get_features_wide([self.intervention_dataset])
            original_column = dg.wide[self.intervention_dataset][
                str(year)
            ].values.reshape(-1, 1)

        max = original_column.max()

        intervened_original = intervened_percent * max / 100

        scaler = StandardScaler()
        scaler.fit(original_column)

        intervened_scaled = scaler.transform(intervened_original.reshape(-1, 1))
        intervened_transformed = sigmoid(intervened_scaled, scale=1 / 3)

        # TODO this output is a bit verbose
        # consider deleting what ends up not needed in the frontend
        percent_calcs = {
            "max": max,
            "intervened_percent": intervened_percent,
            "intervened_original": intervened_original,
            "intervened_scaled": intervened_scaled[0, 0],
            "intervened_transformed": intervened_transformed[0, 0],
        }

        return percent_calcs

    def get_intervened_and_observed_values_original_scale(
        self, fips, intervened_value, year
    ):
        dg = DataGrabber()
        dg.get_features_std_wide([self.intervention_dataset, self.outcome_dataset])
        dg.get_features_wide([self.intervention_dataset])

        # intervened value, in the original scale
        intervened_original_scale = revert_standardize_and_scale_scaler(
            intervened_value, year, self.intervention_dataset
        )

        fips_id = (
            dg.std_wide[self.intervention_dataset]
            .loc[dg.std_wide[self.intervention_dataset]["GeoFIPS"] == fips]
            .index[0]
        )

        # observed value, in the original scale
        observed_original_scale = dg.wide[self.intervention_dataset].iloc[fips_id][
            str(year)
        ]

        return (intervened_original_scale[0], observed_original_scale)

    def get_group_predictions(self, group, intervened_value, year = None,
                              intervention_is_percentile=False):
        self.group = group
        
        if self.data is None:
            file_path = os.path.join(
                self.root,
                "data/years_available",
                f"{self.intervention_dataset}_{self.outcome_dataset}.pkl",
            )
            with open(file_path, "rb") as file:
                self.data = dill.load(file)

        if year is None:
            year = self.data["years_available"][-1]
            assert year in self.data["years_available"]

        self.year = year

        if intervention_is_percentile:
            self.intervened_percentile = intervened_value
            intervened_value = transformed_intervention_from_percentile(
                self.intervention_dataset, year, intervened_value
            )

        self.intervened_value = intervened_value

        # find years for prediction
        outcome_years = self.data["outcome_years"]
        year_id = [int(x) for x in outcome_years].index(year)
        self.year_id = year_id

        self.prediction_years = outcome_years[(year_id) : (year_id + 4)]


        dg = DataGrabber()
        dg.get_features_std_wide([self.intervention_dataset, self.outcome_dataset])
        dg.get_features_wide([self.intervention_dataset])
        interventions_this_year_original = dg.wide[self.intervention_dataset][str(year)]

        self.intervened_value_original = revert_standardize_and_scale_scaler(
            self.intervened_value, self.year, self.intervention_dataset
        )

        self.intervened_value_percentile = round(
            (
                np.mean(
                    interventions_this_year_original.values
                    <= self.intervened_value_original
                )
                * 100
            ),
            3,
        )


        self.fips_ids = (
            dg.std_wide[self.intervention_dataset].
            loc[dg.std_wide[self.intervention_dataset]["GeoFIPS"].isin(self.group)].index.tolist())

        assert len(self.fips_ids) == len(self.group)
        assert list(dg.std_wide[self.intervention_dataset]["GeoFIPS"].iloc[self.fips_ids]) == self.group


        self.names = dg.std_wide[self.intervention_dataset]["GeoName"].iloc[self.fips_ids]

        self.observed_interventions = dg.std_wide[self.intervention_dataset].iloc[
            self.fips_ids
        ][str(year)]

        self.observed_interventions_original = dg.wide[self.intervention_dataset].iloc[
            self.fips_ids
        ][str(year)]

        if intervention_is_percentile:
            self.observed_interventions_percentile =(
                np.round([np.mean(interventions_this_year_original.values <= 
                        obs) for obs in self.observed_interventions_original],3) * 100
            )
            
        self.observed_outcomes = dg.std_wide[self.outcome_dataset].iloc[self.fips_ids][
            outcome_years[year_id : (year_id + 4)]
        ]
        
        self.intervention_diffs = self.intervened_value - self.observed_interventions       
        
        self.intervention_impacts = {}
        self.intervention_impacts_means = []
        self.intervention_impacts_lows = []
        self.intervention_impacts_highs = []
        for shift in [1, 2, 3]:
            self.intervention_impacts[shift] =  np.outer(
               self.tensed_tau_samples[shift], 
                     self.intervention_diffs)
            self.intervention_impacts_means.append(
                np.mean(self.intervention_impacts[shift], axis = 0)
            )
            self.intervention_impacts_lows.append(
                np.percentile(self.intervention_impacts[shift],
                               axis = 0, q= 2.5)
            )
            self.intervention_impacts_highs.append(
                np.percentile(self.intervention_impacts[shift], 
                              axis = 0, q = 97.5)
            )

#self.observed_outcomes.iloc[:,1] * self.intervention_impacts_means[0]
        predicted_mean = self.observed_outcomes.iloc[:,0]




    def get_fips_predictions(
        self, fips, intervened_value, year=None, intervention_is_percentile=False
    ):
        self.fips = fips

        if self.data is None:
            file_path = os.path.join(
                self.root,
                "data/years_available",
                f"{self.intervention_dataset}_{self.outcome_dataset}.pkl",
            )
            with open(file_path, "rb") as file:
                self.data = dill.load(file)

        # start with the latest year possible by default
        if year is None:
            year = self.data["years_available"][-1]
        assert year in self.data["years_available"]

        self.year = year

        if intervention_is_percentile:
            self.intervened_percentile = intervened_value
            intervened_value = transformed_intervention_from_percentile(
                self.intervention_dataset, year, intervened_value
            )

        self.intervened_value = intervened_value

        # find years for prediction
        outcome_years = self.data["outcome_years"]
        year_id = [int(x) for x in outcome_years].index(year)
        self.year_id = year_id

        self.prediction_years = outcome_years[(year_id) : (year_id + 4)]

        
        dg = DataGrabber()
        dg.get_features_std_wide([self.intervention_dataset, self.outcome_dataset])
        dg.get_features_wide([self.intervention_dataset])
        interventions_this_year_original = dg.wide[self.intervention_dataset][str(year)]

        self.intervened_value_original = revert_standardize_and_scale_scaler(
            self.intervened_value, self.year, self.intervention_dataset
        )

        self.intervened_value_percentile = round(
            (
                np.mean(
                    interventions_this_year_original.values
                    <= self.intervened_value_original
                )
                * 100
            ),
            3,
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

        if intervention_is_percentile:
            self.observed_intervention_percentile = round(
                (
                    np.mean(
                        interventions_this_year_original.values
                        <= self.observed_intervention_original
                    )
                    * 100
                ),
                1,
            )

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

        if hasattr(self, "intervened_percentile"):
            intervened_value = self.intervened_percentile
            observed_intervention = self.observed_intervention_percentile

        else:
            intervened_value = round(self.intervened_value, 3)
            observed_intervention = round(self.observed_intervention, 3)

        if scaling == "transformed":
            title = (
                f"Predicted {self.outcome_dataset} in {self.name} under intervention {intervened_value} "
                f"in year {self.year}<br>"
                f"compared to the observed values under observed intervention "
                f"{observed_intervention}."
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
