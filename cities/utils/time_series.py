import torch

def summarize_time_series(
    samples,
    y_true,
    y_site="y_stacked",
    sample_dim=0,
    clamp_at_zero=False,
    compute_metrics=True,
):

    n_series = y_true.shape[-2] if len(y_true.shape) > 1 else 0
    # T = y_true.shape[-1]
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
        total_r2 = 1 - (
            total_model_squared_errors.sum() / total_null_squared_errors.sum()
        )

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