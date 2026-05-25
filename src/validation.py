"""
Walk-forward validation and calibration for the DataGol matchup model.

Walk-forward validation
-----------------------
Respects temporal ordering so there is no data leakage: the model is always
trained on matches played BEFORE the test window, never on future data.

For each 2-week window the function:
  1. Trains a fresh model (via model_factory) on all data before the window.
  2. Predicts on matches inside the window.
  3. Records MAE, RMSE, bias, and whether the model beats the naive baseline.

The naive baseline is: always predict mean(y_train) — if the model cannot
beat this, the personality signal is not statistically useful.

Calibration
-----------
Checks whether predicted xG values are well-calibrated against actual goals.
A calibrated model should have E[actual goals | predicted xG = x] ≈ x.
Miscalibration reveals systematic over/under-prediction at specific ranges.

Plotly output
-------------
All plotting functions return plotly.graph_objects.Figure objects so they
compose directly into a Plotly Dash layout without extra wrappers.
"""

import logging
import warnings
from datetime import timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Plotly is optional at import time; functions that need it raise clearly.
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def baseline_mae(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """
    MAE of the naive predictor that always predicts mean(y_train).
    Any useful model must beat this score.
    """
    prediction = float(np.mean(y_train))
    return float(np.mean(np.abs(y_test - prediction)))


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def walk_forward_validate(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    min_train_samples: int = 50,
    test_window_weeks: int = 2,
    fit_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Evaluate model_factory using expanding-window walk-forward validation.

    Parameters
    ----------
    model_factory : callable
        Called with no arguments; must return a fresh, unfitted model.
        The returned model must expose:
          - .fit(X_train, y_train, **fit_kwargs)
          - .predict(X_test) → array-like of shape (N,) or (N, 1)
        Works with Keras models, sklearn estimators, and anything that
        follows that interface.
    X : np.ndarray, shape (N, 2, 11)
        Lineup arrays.  Axis 2 = [GK, 10 outfield], values = category IDs.
    y : np.ndarray, shape (N,)
        Goals scored by the home team per lineup snapshot.
    dates : np.ndarray, shape (N,), dtype object
        ISO date strings ('YYYY-MM-DD') aligned with X and y rows.
        Empty strings are treated as unknown dates and excluded.
    min_train_samples : int
        Skip windows where fewer than this many training samples are available.
    test_window_weeks : int
        Width of each test window in weeks (default 2).
    fit_kwargs : dict | None
        Extra keyword arguments passed to model.fit().
        For Keras: {'epochs': 60, 'batch_size': 64, 'verbose': 0}.
        For sklearn: {} (empty).

    Returns
    -------
    pd.DataFrame with one row per test window and columns:
        window_start, window_end, n_train, n_test,
        mae, rmse, bias, baseline_mae, beats_baseline,
        mean_pred, mean_actual
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    # -- Parse and sort by date --------------------------------------------
    parsed_dates = _parse_dates(dates)
    valid_mask = ~pd.isnull(parsed_dates)
    if valid_mask.sum() == 0:
        logger.warning("walk_forward_validate: no valid dates found; cannot split temporally.")
        return pd.DataFrame()

    order = np.argsort(parsed_dates[valid_mask])
    X_s = X[valid_mask][order]
    y_s = y[valid_mask][order]
    d_s = parsed_dates[valid_mask][order]

    # -- Build test windows ------------------------------------------------
    window_delta = timedelta(weeks=test_window_weeks)
    t_min = d_s[0]
    t_max = d_s[-1]

    records = []
    window_start = t_min

    while window_start < t_max:
        window_end = window_start + window_delta

        train_mask = d_s < window_start
        test_mask = (d_s >= window_start) & (d_s < window_end)

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        if n_train < min_train_samples or n_test == 0:
            window_start = window_end
            continue

        X_train, y_train = X_s[train_mask], y_s[train_mask]
        X_test, y_test = X_s[test_mask], y_s[test_mask]

        try:
            model = model_factory()
            model.fit(X_train, y_train, **fit_kwargs)
            y_pred = np.asarray(model.predict(X_test)).ravel()
        except Exception as exc:
            logger.warning("Window %s–%s: model failed (%s); skipping.", window_start, window_end, exc)
            window_start = window_end
            continue

        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        bias = float(np.mean(y_pred - y_test))
        base = baseline_mae(y_train, y_test)

        records.append({
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end": window_end.strftime("%Y-%m-%d"),
            "n_train": n_train,
            "n_test": n_test,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "baseline_mae": round(base, 4),
            "beats_baseline": mae < base,
            "mean_pred": round(float(np.mean(y_pred)), 4),
            "mean_actual": round(float(np.mean(y_test)), 4),
        })

        window_start = window_end

    results = pd.DataFrame(records)
    if not results.empty:
        wins = results["beats_baseline"].sum()
        total = len(results)
        logger.info(
            "Walk-forward: %d windows, model beats baseline in %d/%d (%.0f%%).",
            total, wins, total, 100 * wins / total,
        )
    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for xG predictions vs actual goals.

    A well-calibrated model has mean(actual) ≈ mean(predicted) in every
    prediction decile.  Systematic gaps reveal over/under-prediction.

    Parameters
    ----------
    y_true : np.ndarray  actual goals scored (non-negative integers)
    y_pred : np.ndarray  predicted expected goals (positive floats)
    n_bins : int         number of calibration bins (default 10 = deciles)

    Returns
    -------
    dict with keys:
        mae          : float
        rmse         : float
        bias         : float   (positive = model over-predicts)
        r2           : float   (coefficient of determination; 1.0 = perfect)
        bins         : pd.DataFrame  columns [bin_centre, mean_pred, mean_actual, n]
        summary      : str    human-readable one-liner
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float(np.mean(y_pred - y_true))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Calibration bins
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # remove duplicates for small datasets
    bin_ids = np.digitize(y_pred, bin_edges[1:-1])

    bin_records = []
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_records.append({
            "bin_centre": round((bin_edges[b] + bin_edges[b + 1]) / 2, 3),
            "mean_pred": round(float(np.mean(y_pred[mask])), 3),
            "mean_actual": round(float(np.mean(y_true[mask])), 3),
            "n": int(mask.sum()),
        })

    direction = "over-predicts" if bias > 0.05 else ("under-predicts" if bias < -0.05 else "well-calibrated")
    summary = (
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}  R²={r2:.3f}  "
        f"→ model {direction}"
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "r2": round(r2, 4),
        "bins": pd.DataFrame(bin_records),
        "summary": summary,
    }


def compare_vs_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    """
    Side-by-side comparison: model vs naive baseline.

    Returns a DataFrame with index ['model', 'baseline'] and
    columns [mae, rmse, bias, r2, beats_baseline].
    """
    model_report = calibration_report(y_true, y_pred)
    baseline_pred = np.full_like(y_true, fill_value=float(np.mean(y_train)), dtype=float)
    base_report = calibration_report(y_true, baseline_pred)

    return pd.DataFrame(
        {
            "mae":            [model_report["mae"],  base_report["mae"]],
            "rmse":           [model_report["rmse"], base_report["rmse"]],
            "bias":           [model_report["bias"], base_report["bias"]],
            "r2":             [model_report["r2"],   base_report["r2"]],
            "beats_baseline": [model_report["mae"] < base_report["mae"], False],
        },
        index=["model", "baseline"],
    )


# ---------------------------------------------------------------------------
# Plotly figures (Dash-compatible)
# ---------------------------------------------------------------------------


def plot_walk_forward_results(results: pd.DataFrame) -> "go.Figure":
    """
    Interactive Plotly figure with two panels:
      Top   : model MAE vs baseline MAE per window (line chart)
      Bottom: number of test samples per window (bar chart)

    The figure is returned as a go.Figure and can be used directly in a
    Dash layout as:  dcc.Graph(figure=plot_walk_forward_results(results))

    Parameters
    ----------
    results : pd.DataFrame  output of walk_forward_validate()
    """
    _require_plotly()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["MAE vs Baseline", "Test-set size per window"],
        vertical_spacing=0.08,
    )

    windows = results["window_start"].astype(str)
    wins_mask = results["beats_baseline"]

    fig.add_trace(
        go.Scatter(
            x=windows, y=results["mae"],
            mode="lines+markers",
            name="Model MAE",
            line=dict(color="#1f77b4", width=2),
            marker=dict(
                color=["#2ca02c" if v else "#d62728" for v in wins_mask],
                size=9,
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=windows, y=results["baseline_mae"],
            mode="lines",
            name="Baseline MAE",
            line=dict(color="#aec7e8", width=1.5, dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=windows, y=results["n_test"],
            name="Test samples",
            marker_color="#7f7f7f",
            opacity=0.6,
        ),
        row=2, col=1,
    )

    beats_pct = 100 * wins_mask.mean()
    fig.update_layout(
        title=dict(
            text=f"Walk-forward validation  ·  model beats baseline in "
                 f"{wins_mask.sum()}/{len(results)} windows ({beats_pct:.0f}%)",
            font=dict(size=14),
        ),
        height=500,
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="MAE (goals)", row=1, col=1)
    fig.update_yaxes(title_text="N samples", row=2, col=1)
    fig.update_xaxes(title_text="Window start", row=2, col=1)
    return fig


def plot_calibration(report: dict) -> "go.Figure":
    """
    Calibration plot: mean predicted xG vs mean actual goals per decile.

    A perfectly calibrated model lies on the diagonal (y = x).
    Points above the diagonal = under-prediction; below = over-prediction.

    Parameters
    ----------
    report : dict  output of calibration_report()
    """
    _require_plotly()

    bins = report["bins"]
    fig = go.Figure()

    # Ideal calibration line
    max_val = max(bins["mean_pred"].max(), bins["mean_actual"].max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#aec7e8", dash="dash", width=1.5),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bins["mean_pred"],
            y=bins["mean_actual"],
            mode="markers+lines",
            name="Model",
            marker=dict(size=bins["n"] / bins["n"].max() * 20 + 6, color="#1f77b4", opacity=0.8),
            text=[f"n={n}" for n in bins["n"]],
            hovertemplate="Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>%{text}<extra></extra>",
            line=dict(color="#1f77b4", width=1.5),
        )
    )

    fig.update_layout(
        title=dict(text=f"Calibration · {report['summary']}", font=dict(size=13)),
        xaxis_title="Mean predicted xG",
        yaxis_title="Mean actual goals",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def plot_prediction_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> "go.Figure":
    """
    Histogram of prediction errors (y_pred - y_true) with a normal-distribution
    overlay to reveal systematic bias or fat tails.
    """
    _require_plotly()

    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mu, sigma = float(errors.mean()), float(errors.std())

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            name="Prediction error",
            histnorm="probability density",
            marker_color="#1f77b4",
            opacity=0.7,
        )
    )

    x_range = np.linspace(errors.min(), errors.max(), 200)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    fig.add_trace(
        go.Scatter(
            x=x_range, y=normal_pdf,
            mode="lines",
            name=f"Normal(μ={mu:.3f}, σ={sigma:.3f})",
            line=dict(color="#d62728", width=2),
        )
    )

    fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="zero error")

    fig.update_layout(
        title="Prediction error distribution",
        xaxis_title="Error (predicted − actual goals)",
        yaxis_title="Density",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_dates(dates: np.ndarray) -> np.ndarray:
    """Convert an array of ISO strings / datetime-like objects to numpy datetime64."""
    parsed = np.empty(len(dates), dtype="datetime64[D]")
    for i, d in enumerate(dates):
        try:
            if d and str(d) not in ("", "nan", "None"):
                parsed[i] = np.datetime64(pd.to_datetime(str(d)).date(), "D")
            else:
                parsed[i] = np.datetime64("NaT", "D")
        except Exception:
            parsed[i] = np.datetime64("NaT", "D")
    return parsed


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        )
