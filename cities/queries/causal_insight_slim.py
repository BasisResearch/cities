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

    def get_group_predictions(
        self,
        group,
        intervened_value,
        year=None,
        intervention_is_percentile=False,
        produce_original=True,
    ):
        self.group_clean = list(set(group))
        self.group_clean.sort()
        self.produce_original = produce_original

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

        # note: ids will be inceasingly sorted
        self.fips_ids = (
            dg.std_wide[self.intervention_dataset]
            .loc[
                dg.std_wide[self.intervention_dataset]["GeoFIPS"].isin(self.group_clean)
            ]
            .index.tolist()
        )

        assert len(self.fips_ids) == len(self.group_clean)
        assert set(
            dg.std_wide[self.intervention_dataset]["GeoFIPS"].iloc[self.fips_ids]
        ) == set(self.group_clean)

        self.names = dg.std_wide[self.intervention_dataset]["GeoName"].iloc[
            self.fips_ids
        ]

        self.observed_interventions = dg.std_wide[self.intervention_dataset].iloc[
            self.fips_ids
        ][str(year)]

        self.observed_interventions_original = dg.wide[self.intervention_dataset].iloc[
            self.fips_ids
        ][str(year)]

        if intervention_is_percentile:
            self.observed_interventions_percentile = (
                np.round(
                    [
                        np.mean(interventions_this_year_original.values <= obs)
                        for obs in self.observed_interventions_original
                    ],
                    3,
                )
                * 100
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
            self.intervention_impacts[shift] = np.outer(
                self.tensed_tau_samples[shift], self.intervention_diffs
            )
            self.intervention_impacts_means.append(
                np.mean(self.intervention_impacts[shift], axis=0)
            )
            self.intervention_impacts_lows.append(
                np.percentile(self.intervention_impacts[shift], axis=0, q=2.5)
            )
            self.intervention_impacts_highs.append(
                np.percentile(self.intervention_impacts[shift], axis=0, q=97.5)
            )

        intervention_impacts_means_array = np.column_stack(
            self.intervention_impacts_means
        )
        intervention_impacts_lows_array = np.column_stack(
            self.intervention_impacts_lows
        )
        intervention_impacts_highs_array = np.column_stack(
            self.intervention_impacts_highs
        )

        future_predicted_means = (
            self.observed_outcomes.iloc[:, 1:] + intervention_impacts_means_array
        )
        #predicted_means = np.insert(
        #    future_predicted_means, 0, self.observed_outcomes.iloc[:, 0], axis=1
        #) #TODO delete if the new version raises no index error
        
        predicted_means = np.column_stack([self.observed_outcomes.iloc[:,0], future_predicted_means])
        
        

        future_predicted_lows = (
            self.observed_outcomes.iloc[:, 1:] + intervention_impacts_lows_array
        )
        predicted_lows = np.column_stack(
            [self.observed_outcomes.iloc[:, 0], future_predicted_lows]
        )
        # predicted_lows = np.insert(
        #     future_predicted_lows, 0, self.observed_outcomes.iloc[:, 0], axis=1
        # ) #TODO as above

        future_predicted_highs = (
            self.observed_outcomes.iloc[:, 1:] + intervention_impacts_highs_array
        )
        # predicted_highs = np.insert(
        #     future_predicted_highs, 0, self.observed_outcomes.iloc[:, 0], axis=1
        # ) #TODO as above
        
        predicted_highs = np.column_stack(
            [self.observed_outcomes.iloc[:, 0], future_predicted_highs]
        )


        if self.produce_original:
            pred_means_original = []
            pred_lows_original = []
            pred_highs_original = []
            observed_outcomes_original = []
            for i in range(predicted_means.shape[1]):
                y = self.prediction_years[i]
                observed_outcomes_original.append(
                    revert_standardize_and_scale_scaler(
                        self.observed_outcomes.iloc[:, i], y, self.outcome_dataset
                    )
                )
                pred_means_original.append(
                    revert_standardize_and_scale_scaler(
                        predicted_means[:, i], y, self.outcome_dataset
                    )
                )

                pred_lows_original.append(
                    revert_standardize_and_scale_scaler(
                        predicted_lows[:, i], y, self.outcome_dataset
                    )
                )

                pred_highs_original.append(
                    revert_standardize_and_scale_scaler(
                        predicted_highs[:, i], y, self.outcome_dataset
                    )
                )

            pred_means_original = np.column_stack(pred_means_original)
            pred_lows_original = np.column_stack(pred_lows_original)
            pred_highs_original = np.column_stack(pred_highs_original)
            observed_outcomes_original = np.column_stack(observed_outcomes_original)

            self.observed_outcomes_original = pd.DataFrame(observed_outcomes_original)
            self.observed_outcomes_original.index = self.observed_outcomes.index

            assert predicted_means.shape == pred_means_original.shape
            assert predicted_lows.shape == pred_lows_original.shape
            assert predicted_highs.shape == pred_highs_original.shape

        assert int(predicted_means.shape[0]) == len(self.group_clean)
        assert int(predicted_means.shape[1]) == 4
        assert int(predicted_lows.shape[0]) == len(self.group_clean)
        assert int(predicted_lows.shape[1]) == 4
        assert int(predicted_highs.shape[0]) == len(self.group_clean)
        assert int(predicted_highs.shape[1]) == 4

        self.group_predictions = {
            self.group_clean[i]: pd.DataFrame(
                {
                    "year": self.prediction_years,
                    "observed": self.observed_outcomes.loc[self.fips_ids[i]],
                    "mean": predicted_means[i,],
                    "low": predicted_lows[i,],
                    "high": predicted_highs[i,],
                }
            )
            for i in range(len(self.group_clean))
        }

        if self.produce_original:
            self.group_predictions_original = {
                self.group_clean[i]: pd.DataFrame(
                    {
                        "year": self.prediction_years,
                        "observed": self.observed_outcomes_original.loc[
                            self.fips_ids[i]
                        ],
                        "mean": pred_means_original[i,],
                        "low": pred_lows_original[i,],
                        "high": pred_highs_original[i,],
                    }
                )
                for i in range(len(self.group_clean))
            }

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
