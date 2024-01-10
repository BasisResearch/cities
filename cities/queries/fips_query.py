import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cities.utils.data_grabber import (
    DataGrabber,
    MSADataGrabber,
    list_available_features,
    check_if_tensed, 
)
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
        outcome_var=None,
        feature_groups_with_weights=None,
        lag=0,
        top=5,
        time_decay=1.08,
        outcome_comparison_period=None,
        outcome_percentile_range=None,
    ):
        if feature_groups_with_weights is None and outcome_var:
            feature_groups_with_weights = {outcome_var: 4}

        if outcome_var:
            outcome_var_dict = {
                outcome_var: feature_groups_with_weights.pop(outcome_var)
            }
            outcome_var_dict.update(feature_groups_with_weights)
            feature_groups_with_weights = outcome_var_dict

        assert not (
            lag > 0 and outcome_var is None
        ), "Lag will be idle with no outcome variable"

        assert not (
            lag > 0 and outcome_comparison_period is not None
        ), "outcome_comparison_period is only used when lag = 0"

        assert not (
            outcome_var is None and outcome_comparison_period is not None
        ), "outcome_comparison_period requires an outcome variable"

        assert not (
            outcome_var is None and outcome_percentile_range is not None
        ), "outcome_percentile_range requires an outcome variable"

        self.all_available_features = list_available_features()

        feature_groups = list(feature_groups_with_weights.keys())

        assert feature_groups, "You need to specify at least one feature group"

        assert all(
            isinstance(value, int) and -4 <= value <= 4
            for value in feature_groups_with_weights.values()
        ), "Feature weights need to be integers between -4 and 4"

        self.feature_groups_with_weights = feature_groups_with_weights
        self.feature_groups = feature_groups
        self.data = DataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.gdp_var = "gdp"

        # it's fine if they're None (by default)
        self.outcome_var = outcome_var
        self.outcome_comparison_period = outcome_comparison_period

        self.time_decay = time_decay

        if self.gdp_var not in self.feature_groups:
            self.all_features = [self.gdp_var] + feature_groups
        else:
            self.all_features = feature_groups

        self.data.get_features_std_wide(self.all_features)
        self.data.get_features_wide(self.all_features)

        assert (
            fips in self.data.std_wide[self.gdp_var]["GeoFIPS"].values
        ), "FIPS not found in the data set."
        self.name = self.data.std_wide[self.gdp_var]["GeoName"][
            self.data.std_wide[self.gdp_var]["GeoFIPS"] == self.fips
        ].values[0]

        assert (
            self.lag >= 0 and self.lag < 6 and isinstance(self.lag, int)
        ), "lag must be  an iteger between 0 and 5"
        assert (
            self.top > 0
            and isinstance(self.top, int)
            and self.top
            < 2800  # TODO Make sure the number makes sense once we add all datasets we need
        ), "top must be a positive integer smaller than the number of locations in the dataset"

        if outcome_var:
            assert check_if_tensed(
                self.data.std_wide[self.outcome_var]
            ), "Outcome needs to be a time series."

            self.outcome_with_percentiles = self.data.std_wide[self.outcome_var].copy()
            most_recent_outcome = self.data.wide[self.outcome_var].iloc[:, -1].values
            self.outcome_with_percentiles["percentile"] = (
                most_recent_outcome < most_recent_outcome[:, np.newaxis]
            ).sum(axis=1) / most_recent_outcome.shape[0]
            self.outcome_with_percentiles["percentile"] = round(
                self.outcome_with_percentiles["percentile"] * 100, 2
            )
            self.outcome_percentile_range = outcome_percentile_range

    def compare_my_outcome_to_others(self, range_multiplier=2, sample_size=250):
        # TODO add shading by population and warning about
        # locations with low population

        assert self.outcome_var, "Outcome comparison requires an outcome variable."

        self.data.get_features_long([self.outcome_var])
        plot_data = self.data.long[self.outcome_var]
        my_plot_data = plot_data[plot_data["GeoFIPS"] == self.fips].copy()
        my_percentile = self.outcome_with_percentiles["percentile"][
            self.outcome_with_percentiles["GeoFIPS"] == self.fips
        ].values[0]

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

        label_x = my_plot_data["Year"].iloc[-1] - 2
        label_y = my_plot_data["Value"].iloc[-1] * 1.2
        fig.add_annotation(
            text=f"Location recent percentile: {my_percentile}%",
            x=label_x,
            y=label_y,
            showarrow=False,
            font=dict(size=12, color="darkred"),
        )

        title = f"{self.outcome_var} of {self.name}, compared to {sample_size} random other locations"
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=f"{self.outcome_var}",
            template="simple_white",
        )

        fig.show()

    def find_euclidean_kins(self):
        # cut the relevant years from the outcome variable
        if self.outcome_comparison_period and self.outcome_var:
            start_year, end_year = self.outcome_comparison_period

            outcome_df = self.data.std_wide[self.outcome_var].copy()

            condition = (outcome_df.columns[2:].copy().astype(int) >= start_year) & (
                outcome_df.columns[2:].copy().astype(int) <= end_year
            )
            selected_columns = outcome_df.columns[2:][condition].copy()
            filtered_dataframe = outcome_df[selected_columns]

            restricted_df = pd.concat(
                [outcome_df.iloc[:, :2].copy(), filtered_dataframe], axis=1
            )

        elif self.outcome_var:
            restricted_df = self.data.std_wide[self.outcome_var].copy()

        if self.outcome_var:
            self.restricted_outcome_df = restricted_df

        # apply lag in different directions to you and other locations
        # to the outcome variable
        if self.outcome_var:
            self.outcome_slices = slice_with_lag(restricted_df, self.fips, self.lag)

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
        else:
            self.my_df = pd.DataFrame(
                self.data.wide[self.gdp_var][
                    self.data.wide[self.gdp_var]["GeoFIPS"] == self.fips
                ].iloc[:, :2]
            )
            self.other_df = pd.DataFrame(
                self.data.wide[self.gdp_var][
                    self.data.wide[self.gdp_var]["GeoFIPS"] != self.fips
                ].iloc[:, :2]
            )

        # add data on other features to the arrays
        # prior to distance computation

        if self.outcome_var:
            before_shape = self.other_df.shape

        my_features_arrays = np.array([])
        others_features_arrays = np.array([])
        feature_column_count = 0
        for feature in self.feature_groups:
            if feature != self.outcome_var:
                _extracted_df = self.data.wide[feature].copy()
                feature_column_count += _extracted_df.shape[1] - 2
                _extracted_my_df = _extracted_df[_extracted_df["GeoFIPS"] == self.fips]
                _extracted_other_df = _extracted_df[
                    _extracted_df["GeoFIPS"] != self.fips
                ]

                _extracted_other_df.columns = [
                    f"{col}_{feature}" if col not in ["GeoFIPS", "GeoName"] else col
                    for col in _extracted_other_df.columns
                ]

                _extracted_my_df.columns = [
                    f"{col}_{feature}" if col not in ["GeoFIPS", "GeoName"] else col
                    for col in _extracted_my_df.columns
                ]

                assert (
                    _extracted_df.shape[1]
                    == _extracted_my_df.shape[1]
                    == _extracted_other_df.shape[1]
                )

                self.my_df = pd.concat(
                    (self.my_df, _extracted_my_df.iloc[:, 2:]), axis=1
                )

                self.other_df = pd.concat(
                    (self.other_df, _extracted_other_df.iloc[:, 2:]), axis=1
                )

                if self.outcome_var is None:
                    assert (
                        self.my_df.shape[1]
                        == self.other_df.shape[1]
                        == feature_column_count + 2
                    )

                if self.outcome_var:
                    after_shape = self.other_df.shape
                    assert (
                        before_shape[0] == after_shape[0]
                    ), "Feature merging went wrong!"

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

        if len(self.feature_groups) > 1 and self.outcome_var:
            self.my_array = np.hstack((self.my_array, my_features_arrays))
            self.other_arrays = np.hstack((self.other_arrays, others_features_arrays))
        elif self.outcome_var is None:
            self.my_array = my_features_arrays.copy()
            self.other_arrays = others_features_arrays.copy()

        if self.outcome_var is None:
            assert (
                feature_column_count
                == self.my_array.shape[1]
                == self.other_arrays.shape[1]
            )
            assert my_features_arrays.shape == self.my_array.shape
            assert others_features_arrays.shape == self.other_arrays.shape

        compute_weight_array(self, self.time_decay)

        diff = self.all_weights.shape[0] - self.other_arrays.shape[1]
        self.all_weights = self.all_weights[diff:]

        # if self.outcome_var:
        assert (
            self.other_arrays.shape[1] == self.all_weights.shape[0]
        ), "Weights and arrays are misaligned"

        distances = []
        featurewise_contributions = []
        for vector in self.other_arrays:
            _ge = generalized_euclidean_distance(
                np.squeeze(self.my_array), vector, self.all_weights
            )
            distances.append(_ge["distance"])
            featurewise_contributions.append(_ge["featurewise_contributions"])

        # keep weighted distance contribution of each individual feature
        featurewise_contributions_array = np.vstack(featurewise_contributions)

        assert featurewise_contributions_array.shape[1] == len(self.all_weights)

        # turn into df, add ID columns and sort by distance
        featurewise_contributions_df = pd.DataFrame(
            featurewise_contributions_array, columns=self.all_columns
        )
        featurewise_contributions_df[f"distance to {self.fips}"] = distances
        featurewise_contributions_df = pd.concat(
            [self.other_df[["GeoFIPS", "GeoName"]], featurewise_contributions_df],
            axis=1,
        )
        featurewise_contributions_df.sort_values(
            by=featurewise_contributions_df.columns[-1], inplace=True
        )

        # isolate ID columns with distance, tensed columns, atemporal columns
        tensed_column_names = [
            col for col in featurewise_contributions_df.columns if col[:4].isdigit()
        ]
        atemporal_column_names = [
            col for col in featurewise_contributions_df.columns if not col[:4].isdigit()
        ]
        id_column_names = atemporal_column_names[0:2] + [atemporal_column_names[-1]]
        atemporal_column_names = [
            col for col in atemporal_column_names if col not in id_column_names
        ]

        id_df = featurewise_contributions_df[id_column_names]
        tensed_featurewise_contributions_df = featurewise_contributions_df[
            tensed_column_names
        ]
        atemporal_featurewise_contributions_df = featurewise_contributions_df[
            atemporal_column_names
        ]

        # aggregate tensed features (sum across years)
        aggregated_tensed_featurewise_contributions_df = (
            tensed_featurewise_contributions_df.T.groupby(
                tensed_featurewise_contributions_df.columns.str[5:]
            )
            .sum()
            .T
        )

        # aggregate atemporal features (sum across official feature list)
        atemporal_aggregated_dict = {}
        for feature in list(self.all_available_features):
            _selected = [
                col
                for col in atemporal_featurewise_contributions_df.columns
                if col.endswith(feature)
            ]
            if _selected:
                atemporal_aggregated_dict[
                    feature
                ] = atemporal_featurewise_contributions_df[_selected].sum(axis=1)

        aggregated_atemporal_featurewise_contributions_df = pd.DataFrame(
            atemporal_aggregated_dict
        )

        self.featurewise_contributions = featurewise_contributions_df

        # put together the aggregated featurewise contributions
        # and normalize row-wise
        # numbers now mean: "percentage of contribution to the distance"
        self.aggregated_featurewise_contributions = pd.concat(
            [
                id_df,
                aggregated_tensed_featurewise_contributions_df,
                aggregated_atemporal_featurewise_contributions_df,
            ],
            axis=1,
        )
        columns_to_normalize = self.aggregated_featurewise_contributions.iloc[:, 3:]
        self.aggregated_featurewise_contributions.iloc[
            :, 3:
        ] = columns_to_normalize.div(columns_to_normalize.sum(axis=1), axis=0)

        # some sanity checks
        count = sum([1 for distance in distances if distance == 0])

        assert (
            len(distances) == self.other_arrays.shape[0]
        ), "Distances and arrays are misaligned"
        assert (
            len(distances) == self.other_df.shape[0]
        ), "Distances and df are misaligned"

        # #self.other_df[f"distance to {self.fips}"] = distances #remove soon if no errors
        self.other_df.loc[:, f"distance to {self.fips}"] = distances

        count_zeros = (self.other_df[f"distance to {self.fips}"] == 0).sum()
        assert count_zeros == count, "f{count_zeros} zeros in alien distances!"

        # sort and put together euclidean kins
        self.other_df.sort_values(by=self.other_df.columns[-1], inplace=True)

        self.my_df[f"distance to {self.fips}"] = 0

        self.euclidean_kins = pd.concat((self.my_df, self.other_df), axis=0)

        if self.outcome_var:
            self.euclidean_kins = self.euclidean_kins.merge(
                self.outcome_with_percentiles[["GeoFIPS", "percentile"]],
                on="GeoFIPS",
                how="left",
            )

        if self.outcome_var and self.outcome_percentile_range is not None:
            myself = self.euclidean_kins.iloc[:1]
            self.euclidean_kins = self.euclidean_kins[
                self.euclidean_kins["percentile"] >= self.outcome_percentile_range[0]
            ]
            self.euclidean_kins = self.euclidean_kins[
                self.euclidean_kins["percentile"] <= self.outcome_percentile_range[1]
            ]
            self.euclidean_kins = pd.concat([myself, self.euclidean_kins])

    def plot_weights(self):
        plot_weights(self)

    def plot_kins_other_var(self, var, fips_top_custom=None):
        # assert self.outcome_var, "Outcome comparison requires an outcome variable"
        assert hasattr(self, "euclidean_kins"), "Run `find_euclidean_kins` first"

        self.data.get_features_long([var])
        plot_data = self.data.long[var]
        my_plot_data = plot_data[plot_data["GeoFIPS"] == self.fips].copy()
        # up = my_plot_data["Year"].max()
        # possibly remove

        if fips_top_custom is None:
            fips_top = self.euclidean_kins["GeoFIPS"].iloc[1 : (self.top + 1)].values
        else:
            fips_top = fips_top_custom

        others_plot_data = plot_data[plot_data["GeoFIPS"].isin(fips_top)]

        value_column_name = my_plot_data.columns[-1]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=my_plot_data["Year"],
                y=my_plot_data[value_column_name],
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

        for i, fips in enumerate(fips_top):
            subset = others_plot_data[others_plot_data["GeoFIPS"] == fips]
            # line_color = shades_of_grey[i % len(shades_of_grey)]
            line_color = pastel_colors[i % len(pastel_colors)]
            fig.add_trace(
                go.Scatter(
                    x=subset["Year"] + self.lag,
                    y=subset[value_column_name],
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
            yaxis_title=f"{var}",
            legend=dict(title="GeoName"),
            template="simple_white",
        )

        return fig

    def plot_kins(self):
        fig = self.plot_kins_other_var(self.outcome_var)
        return fig

    def show_kins_plot(self):
        fig = self.plot_kins()
        fig.show()


# TODO_Nikodem add population clustering and warning if a population is much different,
# especially if small


class MSAFipsQuery(FipsQuery):
    #     super().__init__(
    #         fips,
    #         outcome_var,
    #         feature_groups_with_weights,
    #         lag,
    #         top,
    #         time_decay,
    #         outcome_comparison_period,
    #         outcome_percentile_range,
    #     )
    def __init__(
        self,
        fips,
        outcome_var=None,
        feature_groups_with_weights=None,
        lag=0,
        top=5,
        time_decay=1.08,
        outcome_comparison_period=None,
        outcome_percentile_range=None,
    ):
        # self.data = MSADataGrabber()
        # self.all_available_features = list_available_features(level="msa")
        # self.gdp_var = "gdp_ma"
        # print("MSAFipsQuery __init__ data:", self.data)

        if feature_groups_with_weights is None and outcome_var:
            feature_groups_with_weights = {outcome_var: 4}

        if outcome_var:
            outcome_var_dict = {
                outcome_var: feature_groups_with_weights.pop(outcome_var)
            }
            outcome_var_dict.update(feature_groups_with_weights)
            feature_groups_with_weights = outcome_var_dict

        assert not (
            lag > 0 and outcome_var is None
        ), "Lag will be idle with no outcome variable"

        assert not (
            lag > 0 and outcome_comparison_period is not None
        ), "outcome_comparison_period is only used when lag = 0"

        assert not (
            outcome_var is None and outcome_comparison_period is not None
        ), "outcome_comparison_period requires an outcome variable"

        assert not (
            outcome_var is None and outcome_percentile_range is not None
        ), "outcome_percentile_range requires an outcome variable"

        self.all_available_features = list_available_features("msa")

        feature_groups = list(feature_groups_with_weights.keys())

        assert feature_groups, "You need to specify at least one feature group"

        assert all(
            isinstance(value, int) and -4 <= value <= 4
            for value in feature_groups_with_weights.values()
        ), "Feature weights need to be integers between -4 and 4"

        self.feature_groups_with_weights = feature_groups_with_weights
        self.feature_groups = feature_groups
        self.data = MSADataGrabber()
        self.repo_root = self.data.repo_root
        self.fips = fips
        self.lag = lag
        self.top = top
        self.gdp_var = "gdp_ma"

        # it's fine if they're None (by default)
        self.outcome_var = outcome_var
        self.outcome_comparison_period = outcome_comparison_period

        self.time_decay = time_decay

        if self.gdp_var not in self.feature_groups:
            self.all_features = [self.gdp_var] + feature_groups
        else:
            self.all_features = feature_groups

        self.data.get_features_std_wide(self.all_features)
        self.data.get_features_wide(self.all_features)

        assert (
            fips in self.data.std_wide[self.gdp_var]["GeoFIPS"].values
        ), "FIPS not found in the data set."
        self.name = self.data.std_wide[self.gdp_var]["GeoName"][
            self.data.std_wide[self.gdp_var]["GeoFIPS"] == self.fips
        ].values[0]

        assert (
            self.lag >= 0 and self.lag < 6 and isinstance(self.lag, int)
        ), "lag must be  an iteger between 0 and 5"
        assert (
            self.top > 0
            and isinstance(self.top, int)
            and self.top
            < 100  # TODO Make sure the number makes sense once we add all datasets we need
        ), "top must be a positive integer smaller than the number of locations in the dataset"

        if outcome_var:
            assert check_if_tensed(
                self.data.std_wide[self.outcome_var]
            ), "Outcome needs to be a time series."

            self.outcome_with_percentiles = self.data.std_wide[self.outcome_var].copy()
            most_recent_outcome = self.data.wide[self.outcome_var].iloc[:, -1].values
            self.outcome_with_percentiles["percentile"] = (
                most_recent_outcome < most_recent_outcome[:, np.newaxis]
            ).sum(axis=1) / most_recent_outcome.shape[0]
            self.outcome_with_percentiles["percentile"] = round(
                self.outcome_with_percentiles["percentile"] * 100, 2
            )
            self.outcome_percentile_range = outcome_percentile_range
