import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict
import seaborn as sns
import math
import textwrap


def summarize_time_series(samples, y_true, y_site="y_stacked", sample_dim=0, clamp_at_zero=False, compute_metrics = True):

    n_series = y_true.shape[-2] if len(y_true.shape) > 1 else 0
    T = y_true.shape[-1]
    summary = {}

    # summarize samples
    mean_pred = samples[y_site].mean(dim=sample_dim).squeeze()
    mean_low = samples[y_site].quantile(0.05, dim=sample_dim).squeeze()
    mean_high = samples[y_site].quantile(0.95, dim=sample_dim).squeeze()

    if clamp_at_zero:
        mean_pred = torch.clamp(mean_pred, min=0)
        mean_low = torch.clamp(mean_low, min=0)
        mean_high = torch.clamp(mean_high, min=0)

    # assert they have the same shapes
    assert mean_pred.shape == y_true.shape
    assert mean_low.shape == y_true.shape

    if compute_metrics:
        # pointwise metrics: global
        total_model_squared_errors = (y_true - mean_pred) ** 2
        total_null_squared_errors = (y_true - y_true.mean()) ** 2

        # single-number metrics: global
        total_rmse = total_model_squared_errors.mean().sqrt()
        total_null_rmse = total_null_squared_errors.mean().sqrt()
        total_r2 = 1 - (total_model_squared_errors.sum() / total_null_squared_errors.sum())

    # summarize series
    series_mean_pred = {}
    series_low_pred = {}
    series_high_pred = {}

    if compute_metrics:
        # pointwise metrics: series
        series_model_squared_errors = {}
        series_null_squared_errors = {}

        # single-number metrics: series
        series_rmse = {}
        series_null_rmse = {}
        series_r2 = {}

    for series in range(n_series):

        # summarize samples
        series_mean_pred[series] = mean_pred[..., series, :]
        series_low_pred[series] = mean_low[..., series, :]
        series_high_pred[series] = mean_high[..., series, :]

        if compute_metrics:
        # pointwise metrics
            series_model_squared_errors[series] = (
                y_true[series, :] - series_mean_pred[series]
            ) ** 2
            series_null_squared_errors[series] = (
                y_true[series, :] - y_true.mean()
            ) ** 2
            series_r2[series] = 1 - (
                series_model_squared_errors[series].sum()
                / series_null_squared_errors[series].sum()
            )
            series_rmse[series] = series_model_squared_errors[series].mean().sqrt()
            series_null_rmse[series] = series_null_squared_errors[series].mean().sqrt()

    summary = {
        "mean_pred": mean_pred,
        "mean_low": mean_low,
        "mean_high": mean_high,
        "series_mean_pred": series_mean_pred,
        "series_low_pred": series_low_pred,
        "series_high_pred": series_high_pred,
        "series_mean_pred": series_mean_pred,
        "series_low_pred": series_low_pred,
        "series_high_pred": series_high_pred,
    }

    if compute_metrics:

        summary = {
            **summary,
            "total_null_se": total_null_squared_errors,
            "total_model_se": total_model_squared_errors,
            "total_r2": total_r2,
            "total_rmse": total_rmse,
            "total_null_rmse": total_null_rmse,
            "series_null_squared_errors": series_null_squared_errors,
            "series_model_squared_errors": series_model_squared_errors,
            "series_null_rmse": series_null_rmse,
            "series_rmse": series_rmse,
            "series_r2": series_r2,
            }

    return summary


def plot_model_summary(summary, y_true, waic=None, title=None, bins = 40, path = None):

    yt = y_true.flatten()
    yt_sorted, indices = torch.sort(yt)

    mean_pred_sorted = summary["mean_pred"].flatten()[indices]
    low_pred_sorted = summary["mean_low"].flatten()[indices]
    high_pred_sorted = summary["mean_high"].flatten()[indices]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    total_r2 = summary["total_r2"]
    total_rmse = summary["total_rmse"]
    main_title = (
        f"Model performance - global R²: {total_r2:.2f}, global RMSE: {total_rmse:.2f} (vs. {summary['total_null_rmse']:.2f} for null model)"
    )
    if waic is not None:
        main_title += f", WAIC: {waic:.2f}"

    main_title += f"\n{title}" if title is not None else ""

    plt.suptitle(main_title, fontsize=16)

    # Plot 1: Predictive Mman vs. true Y
    axs[0, 0].scatter(range(len(yt)), mean_pred_sorted, label="Predictive Mean", s=2, alpha = .5)
    axs[0, 0].scatter(range(len(yt)), yt_sorted, label="True Y", s=2, alpha = 0.5)
    axs[0, 0].fill_between(
        range(len(yt)), low_pred_sorted, high_pred_sorted, color="blue", alpha=0.3
    )
    axs[0, 0].set_title("Predictive Mean vs. True Y (ordered by True Y)")
    axs[0, 0].legend()
    sns.despine(ax=axs[0, 0])

    # Plot 2: squared errors (global)
    axs[0, 1].scatter(summary["total_null_se"], summary["total_model_se"])
    axs[0, 1].set_title("Model vs. Null Squared Errors")
    axs[0, 1].set_xlabel("Null SE")
    axs[0, 1].set_ylabel("Model SE")
    min_se, max_se = torch.min(summary["total_null_se"]), torch.max(
        summary["total_null_se"]
    )
    axs[0, 1].plot([min_se, max_se], [min_se, max_se], linestyle="--", color="red")
    axs[0, 1].set_xlim(min_se, max_se)
    axs[0, 1].set_ylim(min_se, max_se)
    sns.despine(ax=axs[0, 1])

    # Plot 3: R2 across series
    axs[1, 0].hist(
        list(summary["series_r2"].values()), bins=bins, alpha=0.5, color="blue", 

    )
    axs[1, 0].axvline(
        total_r2, color="blue", linestyle="dashed", linewidth=1, label="Dataset R²"
    )
    axs[1, 0].legend()
    axs[1, 0].set_title("R² Distribution Across Series")
    sns.despine(ax=axs[1, 0])

    # Plot 4: RMSE across series
    axs[1, 1].hist(
        list(summary["series_rmse"].values()), bins=bins, alpha=0.5, color="green", 
    )
    axs[1, 1].axvline(
        total_rmse, color="green", linestyle="dashed", linewidth=1, label="Dataset RMSE"
    )
    axs[1, 1].axvline(
        summary["total_null_rmse"],
        color="red",
        linestyle="dashed",
        linewidth=1,
        label="Null Model RMSE",
    )
    axs[1, 1].legend()
    axs[1, 1].set_title("RMSE Distribution Across Series")
    sns.despine(ax=axs[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_coefs(
    samples,
    param_sites,
    # list of random colors of length of param sites
    colors = None,
    true_params=None,
    title=None,
    path = None,
):
    
    if colors is None:
        colors = np.random.rand(len(param_sites), 3) 

    n_sites = len(param_sites)
    n_cols = 2
    n_rows = (n_sites + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes = axes.flatten()

    for i, site in enumerate(param_sites):
        ax = axes[i]
        ax.hist(samples[site].flatten(), bins=30, alpha=0.5, color=colors[i])

        if true_params is not None:
            ax.axvline(
                true_params[i],
                color=colors[i],
                linestyle="dashed",
                linewidth=1,
                label=f"true {site}",
            )

        wrapped_title = "\n".join(textwrap.wrap(f"{site}", width=55))
        ax.set_title(wrapped_title)
        ax.legend()
        sns.despine(ax=ax)

    # Hide any extra subplots if n_sites doesn't match n_rows * n_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Posterior distribution of parameters ({title})", y=1.02)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_selected_series(
    summary,
    intervened_summary = None,
    y_true=None,
    selected_series=None,
    n_series=None,
    series_names=None,
    title=None,
    ylim = None,
    plot_null = False,
    path = None,
    add_metrics = True,
):

    # raise value error if both selected series and n series are passed
    if selected_series is not None and n_series is not None:
        raise ValueError("Only one of selected_series and n_series should be passed.")

    if n_series is None and selected_series is None:
        n_series = y_true.shape[-1]
        selected_series = list(range(n_series))
        print(f"Setting n_series to the maximal value of {n_series}.")

    T = y_true.shape[-1]
    n_selected = len(selected_series)
    n_rows = math.ceil(n_selected / 2)

    fig, axs = plt.subplots(n_rows, 2, figsize=(10, n_rows * 3))

    main_title = "Predictive plots for selected series"
    if title is not None:
        main_title += f" ({title})"

    for i, series in enumerate(selected_series):
        mean_pred = summary["series_mean_pred"][series]
        low_pred = summary["series_low_pred"][series]
        high_pred = summary["series_high_pred"][series]


        if intervened_summary is not None:
            mean_pred_intervened = intervened_summary["series_mean_pred"][series]
            low_pred_intervened = intervened_summary["series_low_pred"][series]
            high_pred_intervened = intervened_summary["series_high_pred"][series]

        ax = axs[i // 2, i % 2] if n_rows > 1 else axs[i % 2]
        if y_true is not None:
            ax.plot(y_true[series, :].detach().numpy(), label="true y")
        ax.plot(mean_pred.detach().numpy(), label="mean prediction")

        if intervened_summary is not None:
            ax.plot(mean_pred_intervened.detach().numpy(), c = "red", label="mean prediction intervened", linestyle="--")
            # ax.fill_between(
            #     range(T), low_pred_intervened.detach().numpy(), high_pred_intervened.detach().numpy(), alpha=0.5, label="90% credible interval intervened"
            # )

        ax.fill_between(
            range(T), low_pred, high_pred, alpha=0.5, label="90% credible interval"
        )

        if plot_null:
            ax.axhline(y_true.mean(), color="gray", linestyle="--", label="null model prediction")

        if add_metrics:
            ax.set_title(
                f"{series_names[series] if series_names is not None else series}, $r^2$ = {summary['series_r2'][series]:.2f}, rmse = {summary['series_rmse'][series]:.2f} (vs. {summary['series_null_rmse'][series]:.2f} for null model)"
            )

        else:
            ax.set_title(f"{series_names[series] if series_names is not None else series}")

        if i == 0:
            ax.legend()

        if ylim is not None:
            ax.set_ylim(ylim)

    sns.despine()
    plt.tight_layout()

    fig.suptitle(main_title, fontsize=16)

    if path is not None:
        plt.savefig(path)
    plt.show()


def plot_ts(
    true_ts,
    title,
    xlabel,
    ylabel,
    other_ts: Dict = {},
    plot_uncertainty: bool = False,
    ax=None,
    legend = True,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    def add_lines_from_tensor(ts, label):
        if len(ts.shape) == 2:
            for i in range(ts.shape[0]):
                ax.plot(ts[i, :], label=f"{label}_{i}")
        else:
            ax.plot(ts, label=label)

    add_lines_from_tensor(true_ts, "true")

    # if len(true_ts.shape) == 2:
    #     for i in range(true_ts.shape[-2]):
    #         plt.plot(true_ts[i,:], label=f'true_{i}')
    # else:
    #     plt.plot(true_ts, label='true')

    if other_ts.keys():
        if not plot_uncertainty:
            for key, ts in other_ts.items():
                add_lines_from_tensor(ts, key)

        else:
            for key, ts in other_ts.items():
                mean = ts.mean(dim=0)
                std = ts.std(dim=0)
                ax.plot(mean, label=key)
                ax.fill_between(
                    np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.2
                )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    sns.despine()

    return fig, ax


# fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # For a 2x2 grid

# # Example usage with each subplot
# plot_ts(true_ts, "Plot 1", "t", "y", ax=axes[0, 0])
# plot_ts(true_ts, "Plot 2", "t", "y", ax=axes[0, 1])
# plot_ts(true_ts, "Plot 3", "t", "y", ax=axes[1, 0])
# plot_ts(true_ts, "Plot 4", "t", "y", ax=axes[1, 1])

# plt.tight_layout()
# plt.show()
