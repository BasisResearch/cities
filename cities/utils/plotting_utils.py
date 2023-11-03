import os

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cities.utils.cleaning_utils import find_repo_root, standardize_and_scale
from cities.utils.data_grabber import (
    DataGrabber,
    list_available_features,
    list_tensed_features,
)


def plot_kins_variable(n_kins, kins_df, variable, type_of_plot="bar"):
    """Plot kins variables in a grid of subplots.
    Args:
        n_kins (int): number of kins to plot

        kins_df (pandas dataframe): dataframe of kins
            To get this, e.g.:
            f  = FipsQuery(fips, outcome_var = "gdp",
               feature_groups_with_weights= {"gdp":0, "population":4},
               lag = 3, top =10, time_decay = 1.03)
            f.find_euclidean_kins()
            kins_df = f.euclidean_kins

        variable (str): variable to plot

        type_of_plot (str): type of plot to use
            options are "bar_multiplot", "stacked_bar_multiplot", "pie", "stacked_bar_singleplot", "bar_singleplot", "line"

    Returns:
        fig (plotly figure): figure
    """
    # load variable data
    path = find_repo_root()
    variable_df = pd.read_csv(
        os.path.join(path, "data/processed/" + variable + "_wide.csv")
    )

    df_kins_variable = get_df_kins_variable(n_kins, kins_df, variable_df)

    match type_of_plot:
        case "bar_multiplot" | "stacked_bar_multiplot" | "pie":  # multiple subplots
            [fig, subplot_row_indices, subplot_col_indices] = grid_of_subplots(
                n_kins, type_of_plot=type_of_plot
            )
            [what_to_plot, trace_row_indices, trace_col_indices] = traces_to_plot(
                df_kins_variable,
                subplot_row_indices,
                subplot_col_indices,
                type_of_plot=type_of_plot,
            )
            fig = place_traces(
                fig,
                trace_row_indices,
                trace_col_indices,
                what_to_plot,
                type_of_plot=type_of_plot,
            )
        case "stacked_bar_singleplot":
            fig = single_plot_bar(df_kins_variable, barmode="stack")
        case "bar_singleplot":
            fig = single_plot_bar(df_kins_variable, barmode="group")
        case "line":
            fig = single_plot_lines(df_kins_variable, variable=variable)

    # general settings
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        # paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def single_plot_lines(df_kins_variable, variable=""):
    fig = go.Figure()
    data_columns = df_kins_variable[
        df_kins_variable.select_dtypes(include=["float64"]).columns
    ]
    for i, row in df_kins_variable.iterrows():
        fig.add_trace(
            go.Scatter(
                x=data_columns.columns,
                y=row[data_columns.columns].values,
                name=row["GeoName"],
                hoverinfo="name+y+x",
            )
        )
    match variable:
        case "gdp":
            axis_label = "GDP"
        case "spending_HHS":
            axis_label = "Health and Human Services Spending"
        case "industry_arts_recreation_total":
            axis_label = "Arts and Recreation Industry"
        case "unemployment_rate":
            axis_label = "Unemployment Rate (%)"
        case "spending_commerce":
            axis_label = "Commerce Spending"
        case "population":
            axis_label = "Population"
        case "spending_transportation":
            axis_label = "Transportation Spending"
        case _:
            axis_label = variable
    fig.update_yaxes(title_text=axis_label)
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis=dict(gridcolor="#E5E5E5"),
        yaxis=dict(gridcolor="#E5E5E5"),
    )
    return fig


def single_plot_bar(df_kins_variable, barmode="stack"):
    fig = go.Figure()
    data_columns = df_kins_variable[
        df_kins_variable.select_dtypes(include=["float64"]).columns
    ]
    for col in data_columns:
        fig.add_trace(
            go.Bar(x=df_kins_variable["GeoName"], y=df_kins_variable[col], name=col)
        )
    fig.update_layout(barmode=barmode)
    fig.update_yaxes(visible=False)
    return fig


def get_df_kins_variable(n_kins, kins_df, variable_df):
    kins_df = kins_df[0:n_kins]
    df_kins_variable = variable_df[variable_df["GeoFIPS"].isin(kins_df["GeoFIPS"])]
    df_kins_variable = df_kins_variable.reset_index(drop=True)
    return df_kins_variable


def grid_of_subplots(n_kins, type_of_plot="bar_multiplot"):
    num_cols = int(np.ceil(np.sqrt(n_kins)))
    num_rows = int(np.ceil(n_kins / num_cols))

    match type_of_plot:
        case "bar_multiplot":
            subplot_type = "bar"
        case "pie":
            subplot_type = "pie"
        case "stacked_bar_multiplot":
            subplot_type = "bar"

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        specs=[
            [{"type": subplot_type} for j in range(num_cols)] for i in range(num_rows)
        ],
    )

    subplot_row_indices = [
        i for i in range(1, num_rows + 1) for j in range(1, num_cols + 1)
    ][:n_kins]

    subplot_col_indices = [
        j for i in range(1, num_rows + 1) for j in range(1, num_cols + 1)
    ][:n_kins]

    return fig, subplot_row_indices, subplot_col_indices


def traces_to_plot(
    df_kins_variable,
    subplot_row_indices,
    subplot_col_indices,
    type_of_plot="bar_multiplot",
):
    match type_of_plot:
        case "bar_multiplot":
            what_to_plot, trace_row_indices, trace_col_indices = traces_to_plot_bar(
                df_kins_variable, subplot_row_indices, subplot_col_indices
            )
        case "pie":
            what_to_plot, trace_row_indices, trace_col_indices = traces_to_plot_pie(
                df_kins_variable, subplot_row_indices, subplot_col_indices
            )
        case "stacked_bar_multiplot":
            what_to_plot, trace_row_indices, trace_col_indices = traces_to_plot_bar(
                df_kins_variable, subplot_row_indices, subplot_col_indices
            )

    return what_to_plot, trace_row_indices, trace_col_indices


def traces_to_plot_pie(df_kins_variable, subplot_row_indices, subplot_col_indices):
    what_to_plot = []
    trace_row_indices = []
    trace_col_indices = []

    # find float columns
    data_columns = df_kins_variable[
        df_kins_variable.select_dtypes(include=["float64"]).columns
    ]

    # Define your color sequence
    color_sequence = plotly.colors.DEFAULT_PLOTLY_COLORS

    for i, row in df_kins_variable.iterrows():
        trace = go.Pie(
            labels=data_columns.columns,
            values=row[data_columns.columns].values,
            name=row["GeoName"],
            showlegend=True,
            textinfo="none",
            marker=dict(colors=color_sequence),
            title=row["GeoName"],
            hoverinfo="percent+label",
            hole=0.6,
        )
        what_to_plot.append(trace)
        trace_row_indices.append(subplot_row_indices[i])
        trace_col_indices.append(subplot_col_indices[i])
    return what_to_plot, trace_row_indices, trace_col_indices


def traces_to_plot_bar(df_kins_variable, subplot_row_indices, subplot_col_indices):
    what_to_plot = []
    trace_row_indices = []
    trace_col_indices = []

    # find float columns
    data_columns = df_kins_variable[
        df_kins_variable.select_dtypes(include=["float64"]).columns
    ]

    # Define your color sequence
    color_sequence = plotly.colors.DEFAULT_PLOTLY_COLORS

    for i, row in df_kins_variable.iterrows():
        color_index = 0  # Reset color index for each subplot
        for col in data_columns:
            trace = go.Bar(
                x=[row["GeoName"]],
                y=[row[col]],
                name=col,
                marker_color=color_sequence[
                    color_index % len(color_sequence)
                ],  # Use modulo to cycle through colors
                hoverinfo="name",
            )
            what_to_plot.append(trace)
            trace_row_indices.append(subplot_row_indices[i])
            trace_col_indices.append(subplot_col_indices[i])
            color_index += 1  # Increment color index
    return what_to_plot, trace_row_indices, trace_col_indices


def place_traces(
    fig,
    trace_row_indices,
    trace_col_indices,
    what_to_plot,
    type_of_plot="bar_multiplot",
):
    # place traces
    for i in range(len(what_to_plot)):
        fig.add_trace(
            what_to_plot[i], row=trace_row_indices[i], col=trace_col_indices[i]
        )

    # type-specific settings
    match type_of_plot:
        case "stacked_bar_multiplot":
            fig.update_layout(barmode="stack")
            fig.update_yaxes(visible=False)
        case "bar_multiplot":
            fig.update_yaxes(visible=False)
    return fig
