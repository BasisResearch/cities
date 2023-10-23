import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cities.utils.data_grabber import DataGrabber
from cities.utils.similarity_utils import (
    compute_weight_array,
    generalized_euclidean_distance,
    plot_weights,
    slice_with_lag,
)

# from scipy.spatial import distance


class FipsQuery:
    def __init__(
        self,
        fips,
        outcome_var="gdp",
        feature_groups_with_weights=None,
        lag=0,
        top=5,
        time_decay=1.08,
    ):
        if feature_groups_with_weights is None:
            feature_groups_with_weights = {outcome_var: 4}

        assert outcome_var in [
            "gdp",
            "population",
        ], "outcome_var must be one of ['gdp', 'population']"
        # TODO_Nikodem update once variable added

        feature_groups = list(feature_groups_with_weights.keys())

        assert all(
            isinstance(value, int) and -4 <= value <= 4
            for value in feature_groups_with_weights.values()
        )

        self.feature_groups_with_weights = feature_groups_with_weights
        self.feature_groups = feature_groups
        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.outcome_var = outcome_var
        self.time_decay = time_decay

        if "gdp" not in self.feature_groups:
            self.all_features = ["gdp"] + feature_groups
        else:
            self.all_features = feature_groups

        self.data.get_features_std_wide(self.all_features)
        self.data.get_features_wide(self.all_features)

        # TODO_Nikodem: dropping columns that are excluded by `how_far_back`

        assert (
            fips in self.data.std_wide["gdp"]["GeoFIPS"].values
        ), "FIPS not found in the data set."
        self.name = self.data.std_wide["gdp"]["GeoName"][
            self.data.std_wide["gdp"]["GeoFIPS"] == self.fips
        ].values[0]

        assert (
            self.lag >= 0 and self.lag < 6 and isinstance(self.lag, int)
        ), "lag must be  an iteger between 0 and 5"
        assert (
            self.top > 0
            and isinstance(self.top, int)
            and self.top < self.data.std_wide[self.outcome_var].shape[0]
        ), "top must be a positive integer smaller than the number of locations in the dataset"

    def compare_my_outcome_to_others(self, range_multiplier=2, sample_size=250):
        # TODO add shading by population and warning about
        # locations with low population

        # TODO consider explicit printing of percentiles in
        # complete data set

        self.data.get_features_long([self.outcome_var])
        plot_data = self.data.long[self.outcome_var]
        my_plot_data = plot_data[plot_data["GeoFIPS"] == self.fips].copy()

        others_plot_data = plot_data[plot_data["GeoFIPS"] != self.fips]

        fips = others_plot_data["GeoFIPS"].unique()
        sampled_fips = np.random.choice(fips, sample_size, replace=False)
        others_sampled_plot_data = plot_data[plot_data["GeoFIPS"].isin(sampled_fips)]

        y_min = my_plot_data["Value"].mean() - (
            range_multiplier * my_plot_data["Value"].std()
        )
        y_max = my_plot_data["Value"].mean() + (
            range_multiplier * my_plot_data["Value"].std()
        )

        fig = go.Figure(layout_yaxis_range=[y_min, y_max])

        for i, geoname in enumerate(others_sampled_plot_data["GeoName"].unique()):
            subset = others_plot_data[others_plot_data["GeoName"] == geoname]
            # line_color = shades_of_grey[i % len(shades_of_grey)]
            # line_color = pastel_colors[i % len(pastel_colors)]
            line_color = "lightgray"
            fig.add_trace(
                go.Scatter(
                    x=subset["Year"],
                    y=subset["Value"],
                    mode="lines",
                    name=subset["GeoName"].iloc[0],
                    line_color=line_color,
                    text=subset["GeoName"].iloc[0],
                    textposition="top right",
                    showlegend=False,
                    opacity=0.4,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=my_plot_data["Year"],
                y=my_plot_data["Value"],
                mode="lines",
                name=my_plot_data["GeoName"].iloc[0],
                line=dict(color="darkred", width=3),
                text=my_plot_data["GeoName"].iloc[0],
                textposition="top right",
                showlegend=False,
            )
        )

        title = f"{self.outcome_var} of {self.name}, compared to {sample_size} random other locations"
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=f"{self.outcome_var}",
            template="simple_white",
        )

        fig.show()

    def find_euclidean_kins(self):  # TODO_Nikodem add a test for this function
        self.outcome_slices = slice_with_lag(
            self.data.std_wide[self.outcome_var], self.fips, self.lag
        )

        self.my_array = np.array(self.outcome_slices["my_array"])
        self.other_arrays = np.array(self.outcome_slices["other_arrays"])

        assert self.my_array.shape[1] == self.other_arrays.shape[1]

        self.my_df = self.data.wide[self.outcome_var][
            self.data.wide[self.outcome_var]["GeoFIPS"] == self.fips
        ].copy()

        self.other_df = self.outcome_slices["other_df"]
        self.other_df = self.data.wide[self.outcome_var][
            self.data.wide[self.outcome_var]["GeoFIPS"] != self.fips
        ].copy()

        # add data on other features listed to the arrays
        # prior to distance computation
        my_features_arrays = np.array([])
        others_features_arrays = np.array([])
        for feature in self.feature_groups:
            if feature != self.outcome_var:
                _extracted_df = self.data.wide[feature].copy()
                _extracted_my_df = _extracted_df[_extracted_df["GeoFIPS"] == self.fips]
                _extracted_other_df = _extracted_df[
                    _extracted_df["GeoFIPS"] != self.fips
                ]

                before_shape = self.other_df.shape

                assert (
                    self.other_df["GeoFIPS"].unique()
                    == _extracted_other_df["GeoFIPS"].unique()
                ).all(), "FIPS are missing"

                assert (
                    self.other_df["GeoFIPS"] == _extracted_other_df["GeoFIPS"]
                ).all(), "FIPS are misaligned"

                _extracted_other_df.columns = [
                    f"{col}_{feature}" if col not in ["GeoFIPS", "GeoName"] else col
                    for col in _extracted_other_df.columns
                ]

                _extracted_my_df.columns = [
                    f"{col}_{feature}" if col not in ["GeoFIPS", "GeoName"] else col
                    for col in _extracted_my_df.columns
                ]

                self.my_df = pd.concat(
                    (self.my_df, _extracted_my_df.iloc[:, 2:]), axis=1
                )
                self.other_df = pd.concat(
                    (self.other_df, _extracted_other_df.iloc[:, 2:]), axis=1
                )

                after_shape = self.other_df.shape

                assert before_shape[0] == after_shape[0], "Feature merging went wrong!"

                _extracted_df_std = self.data.std_wide[feature].copy()
                _extracted_other_array = np.array(
                    _extracted_df_std[_extracted_df_std["GeoFIPS"] != self.fips].iloc[
                        :, 2:
                    ]
                )
                _extracted_my_array = np.array(
                    _extracted_df_std[_extracted_df_std["GeoFIPS"] == self.fips].iloc[
                        :, 2:
                    ]
                )

                if my_features_arrays.size == 0:
                    my_features_arrays = _extracted_my_array
                else:
                    my_features_arrays = np.hstack(
                        (my_features_arrays, _extracted_my_array)
                    )

                if others_features_arrays.size == 0:
                    others_features_arrays = _extracted_other_array
                else:
                    others_features_arrays = np.hstack(
                        (others_features_arrays, _extracted_other_array)
                    )

        if len(self.feature_groups) > 1:
            self.my_array = np.hstack((self.my_array, my_features_arrays))
            self.other_arrays = np.hstack((self.other_arrays, others_features_arrays))

        compute_weight_array(self, self.time_decay)

        diff = self.all_weights.shape[0] - self.other_arrays.shape[1]
        self.all_weights = self.all_weights[diff:]

        assert (
            self.other_arrays.shape[1] == self.all_weights.shape[0]
        ), "Weights and arrays are misaligned"

        distances = []
        for vector in self.other_arrays:
            distances.append(
                generalized_euclidean_distance(
                    np.squeeze(self.my_array), vector, self.all_weights
                )
            )
            # distances.append(
            #     distance.euclidean(
            #         np.squeeze(self.my_array), vector, w=self.all_weights
            #     )
            # )

        count = sum([1 for distance in distances if distance == 0])

        assert (
            len(distances) == self.other_arrays.shape[0]
        ), "Distances and arrays are misaligned"
        assert (
            len(distances) == self.other_df.shape[0]
        ), "Distances and df are misaligned"

        self.other_df[f"distance to {self.fips}"] = distances
        count_zeros = (self.other_df[f"distance to {self.fips}"] == 0).sum()

        assert count_zeros == count, "f{count_zeros} zeros in alien distances!"

        self.other_df.sort_values(by=self.other_df.columns[-1], inplace=True)

        self.my_df[f"distance to {self.fips}"] = 0

        self.euclidean_kins = pd.concat((self.my_df, self.other_df), axis=0)

    def plot_weights(self):
        plot_weights(self)

    def plot_kins(self):
        self.data.get_features_long([self.outcome_var])
        plot_data = self.data.long[self.outcome_var]
        my_plot_data = plot_data[plot_data["GeoFIPS"] == self.fips].copy()
        up = my_plot_data["Year"].max()

        fips_top = self.euclidean_kins["GeoFIPS"].iloc[1 : (self.top + 1)].values
        others_plot_data = plot_data[plot_data["GeoFIPS"].isin(fips_top)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=my_plot_data["Year"],
                y=my_plot_data["Value"],
                mode="lines",
                name=my_plot_data["GeoName"].iloc[0],
                line=dict(color="darkred", width=3),
                text=my_plot_data["GeoName"].iloc[0],
                textposition="top right",
            )
        )

        # TODO_Nikodem maybe add more shades and test on largish top
        # shades_of_grey = ["#333333", "#444444", "#555555", "#666666", "#777777"][: self.top]

        pastel_colors = ["#FFC0CB", "#A9A9A9", "#87CEFA", "#FFD700", "#98FB98"][
            : self.top
        ]

        for i, geoname in enumerate(others_plot_data["GeoName"].unique()):
            subset = others_plot_data[others_plot_data["GeoName"] == geoname]
            # line_color = shades_of_grey[i % len(shades_of_grey)]
            line_color = pastel_colors[i % len(pastel_colors)]
            fig.add_trace(
                go.Scatter(
                    x=subset["Year"] + self.lag,
                    y=subset["Value"],
                    mode="lines",
                    name=subset["GeoName"].iloc[0],
                    line_color=line_color,
                    text=subset["GeoName"].iloc[0],
                    textposition="top right",
                )
            )

        if self.lag > 0:
            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        x0=2021,
                        x1=2021,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="darkgray", width=2),
                    )
                ]
            )

            fig.add_annotation(
                text=f"their year {2021 - self.lag}",
                x=2021.0,
                y=1.05,
                xref="x",
                yref="paper",
                showarrow=False,
                font=dict(color="darkgray"),
            )

        top = self.top
        lag = self.lag
        title_1 = title = f"Top {self.top} locations matching your search"
        title_2 = (
            f"Top {self.top} locations matching your search (lag of {self.lag} years)"
        )

        if not self.feature_groups:
            if self.lag == 0:
                title = title_1
            else:
                title = title_2
        else:
            if self.lag == 0:
                title = f"Top {top} locations matching your search"
            else:
                title = f"Top {top} locations matching your search (lag of {lag} years)"

        # TODO will need to mention how_far_back if we implement it  \
        # TODO adding info about feature cluster weights at the bottom of the plot

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=f"{self.outcome_var}",
            legend=dict(title="GeoName"),
            template="simple_white",
        )

        fig.show()


# TODO_Nikodem add population clustering and warning if a population is much different,
# especially if small
