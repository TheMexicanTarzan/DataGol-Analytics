"""
Backtesting for the DataGol two-layer matchup model.

Inputs expected (all aligned by match index):
  model_probs : pd.DataFrame  columns [home_win, draw, away_win]  (model probabilities)
  odds_df     : pd.DataFrame  columns [date, home_team, away_team,
                                        home_odds, draw_odds, away_odds,
                                        home_goals, away_goals, outcome]
  (outcome is 'H', 'D', or 'A')

The two DataFrames are merged on (home_team, away_team, date) — fuzzy name
matching is used for teams since model and odds sources may use different names.

Kelly Criterion
---------------
For each outcome o with model probability p_o and decimal odds b_o:

  edge_o     = p_o - fair_prob_o          (fair_prob removes bookmaker margin)
  kelly_o    = (b_o * p_o - (1 - p_o)) / b_o   (full Kelly)
  bet_size_o = kelly_fraction * kelly_o * bankroll  (fractional Kelly)

A bet is placed only when edge_o > min_edge (default 0.05).
At most ONE outcome is bet per match (the one with the highest edge).

Performance metrics
-------------------
  roi             : total profit / total staked
  final_bankroll  : ending bankroll value
  max_drawdown    : largest peak-to-trough bankroll decline (%)
  sharpe_ratio    : mean(returns) / std(returns) * sqrt(52)  [weekly]
  win_rate        : fraction of bets won
  n_bets          : total bets placed
  avg_edge        : average edge on placed bets
  avg_kelly       : average Kelly fraction on placed bets
"""

import difflib
import logging
import unicodedata

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Plotly is optional at import time; functions that need it raise clearly.
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _remove_margin(
    home_odds: float, draw_odds: float, away_odds: float
) -> "tuple[float, float, float]":
    """Fair implied probabilities (margin removed).

    Same logic as remove_bookmaker_margin in odds_scraper.py, duplicated here
    to avoid a circular import.
    """
    rp = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    total = sum(rp)
    return tuple(p / total for p in rp)


def _kelly(p: float, b: float) -> float:
    """Full Kelly fraction.

    Parameters
    ----------
    p : model probability for the outcome
    b : decimal odds minus 1  (i.e. net odds, so b = decimal_odds - 1)

    Returns 0 if the calculated fraction is negative (no edge).
    """
    return max(0.0, (b * p - (1 - p)) / b)


def _max_drawdown(bankroll_series: pd.Series) -> float:
    """Peak-to-trough maximum drawdown as a positive percentage."""
    peak = bankroll_series.cummax()
    drawdown = (bankroll_series - peak) / peak
    return float(drawdown.min() * -100)


def _sharpe(pnl_series: pd.Series, periods_per_year: int = 52) -> float:
    """Annualised Sharpe ratio (weekly returns assumed)."""
    if pnl_series.std() == 0:
        return 0.0
    return float(pnl_series.mean() / pnl_series.std() * (periods_per_year ** 0.5))


def _normalise_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.lower().split())


def _require_plotly() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for plotting. Install with: pip install plotly"
        )


def _empty_result(bankroll: float) -> dict:
    """Return a zeroed result dict when there is nothing to backtest."""
    return {
        "roi": 0.0,
        "final_bankroll": bankroll,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "n_bets": 0,
        "avg_edge": 0.0,
        "avg_kelly": 0.0,
        "history": pd.DataFrame(
            columns=[
                "date", "home_team", "away_team", "outcome",
                "bet_on", "edge", "kelly_fraction", "bet_size",
                "pnl", "bankroll", "cumulative_roi",
            ]
        ),
        "summary": "No bets placed (empty input or no matches exceeded the minimum edge).",
    }


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_predictions_with_odds(
    model_probs: pd.DataFrame,
    odds_df: pd.DataFrame,
    name_cutoff: float = 0.75,
) -> pd.DataFrame:
    """Left-merge odds_df with model_probs using fuzzy team-name matching.

    Parameters
    ----------
    model_probs : pd.DataFrame
        Columns [home_win, draw, away_win].  Its index must align row-for-row
        with odds_df (i.e. row 0 of model_probs corresponds to row 0 of
        odds_df).  If they are already aligned, the merge is trivial.
    odds_df : pd.DataFrame
        Output of FootballDataScraper.get_historical_odds().  Must contain
        columns: date, home_team, away_team, home_odds, draw_odds, away_odds,
        home_goals, away_goals, outcome.
    name_cutoff : float
        Minimum similarity threshold for difflib.get_close_matches (0–1).

    Returns
    -------
    pd.DataFrame with all odds_df columns plus home_win, draw, away_win.
    Rows where no probability match was found are dropped.
    """
    required_odds = {"date", "home_team", "away_team", "home_odds", "draw_odds",
                     "away_odds", "outcome"}
    required_probs = {"home_win", "draw", "away_win"}

    if odds_df.empty or model_probs.empty:
        logger.warning("merge_predictions_with_odds: one or both inputs are empty.")
        return pd.DataFrame()

    missing_odds = required_odds - set(odds_df.columns)
    missing_probs = required_probs - set(model_probs.columns)
    if missing_odds:
        raise ValueError(f"odds_df is missing columns: {missing_odds}")
    if missing_probs:
        raise ValueError(f"model_probs is missing columns: {missing_probs}")

    # If the DataFrames share the same index length, treat them as pre-aligned
    # and attach probabilities directly before doing any name-based lookup.
    odds = odds_df.reset_index(drop=True).copy()
    probs = model_probs.reset_index(drop=True).copy()

    if len(odds) == len(probs):
        logger.info(
            "merge_predictions_with_odds: row counts match (%d); using direct alignment.",
            len(odds),
        )
        merged = odds.copy()
        for col in required_probs:
            merged[col] = probs[col].values
        return merged

    # Otherwise do a fuzzy name-join on (date, home_team, away_team).
    # Build a normalised name map from odds_df teams.
    logger.info(
        "merge_predictions_with_odds: row counts differ (%d vs %d); using fuzzy join.",
        len(odds), len(probs),
    )

    if "home_team" not in probs.columns or "away_team" not in probs.columns:
        logger.warning(
            "model_probs has no team name columns; cannot perform fuzzy join. "
            "Provide equal-length DataFrames for direct alignment."
        )
        return pd.DataFrame()

    odds_home_names = [_normalise_name(n) for n in odds["home_team"]]
    odds_away_names = [_normalise_name(n) for n in odds["away_team"]]
    odds_dates = list(odds["date"].astype(str))

    # Build a lookup: normalised_name -> original index in odds_df
    all_odds_teams = set(odds_home_names + odds_away_names)
    all_odds_team_list = list(all_odds_teams)

    prob_home_names = [_normalise_name(n) for n in probs["home_team"]]
    prob_away_names = [_normalise_name(n) for n in probs["away_team"]]
    prob_dates = list(probs["date"].astype(str))

    def _best_match(query: str, choices: list) -> "str | None":
        matches = difflib.get_close_matches(query, choices, n=1, cutoff=name_cutoff)
        return matches[0] if matches else None

    # Build odds index keyed by (date, normed_home, normed_away)
    odds_index: dict[tuple, int] = {}
    for i, (d, h, a) in enumerate(zip(odds_dates, odds_home_names, odds_away_names)):
        odds_index[(d, h, a)] = i

    prob_rows_matched: list[int] = []
    odds_rows_matched: list[int] = []

    for j, (d, ph, pa) in enumerate(zip(prob_dates, prob_home_names, prob_away_names)):
        # Exact match first
        key = (d, ph, pa)
        if key in odds_index:
            prob_rows_matched.append(j)
            odds_rows_matched.append(odds_index[key])
            continue

        # Fuzzy match on team names within the same date
        same_date_home = [h for (dd, h, a) in odds_index if dd == d]
        same_date_away = {(h, a): odds_index[(dd, h, a)] for (dd, h, a) in odds_index if dd == d}

        bh = _best_match(ph, same_date_home)
        if bh is None:
            bh = _best_match(ph, all_odds_team_list)

        if bh is not None:
            same_date_for_home = [a for (h, a) in same_date_away if h == bh]
            ba = _best_match(pa, same_date_for_home)
            if ba is None:
                ba = _best_match(pa, all_odds_team_list)

            if ba is not None and (d, bh, ba) in odds_index:
                prob_rows_matched.append(j)
                odds_rows_matched.append(odds_index[(d, bh, ba)])
                continue

        logger.debug(
            "merge_predictions_with_odds: no match for prob row %d (%s vs %s on %s).",
            j, ph, pa, d,
        )

    if not prob_rows_matched:
        logger.warning("merge_predictions_with_odds: no rows could be matched.")
        return pd.DataFrame()

    matched_odds = odds.iloc[odds_rows_matched].reset_index(drop=True)
    matched_probs = probs.iloc[prob_rows_matched][list(required_probs)].reset_index(drop=True)
    merged = pd.concat([matched_odds, matched_probs], axis=1)

    logger.info(
        "merge_predictions_with_odds: matched %d/%d probability rows to odds.",
        len(merged), len(probs),
    )
    return merged


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


def run_backtest(
    model_probs: pd.DataFrame,
    odds_df: pd.DataFrame,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
) -> dict:
    """Simulate Kelly-criterion betting on historical matches.

    Parameters
    ----------
    model_probs : pd.DataFrame
        Columns [home_win, draw, away_win].  Row-aligned with odds_df or
        containing team/date columns for fuzzy merging.
    odds_df : pd.DataFrame
        Historical odds data from FootballDataScraper.get_historical_odds().
    bankroll : float
        Starting bankroll (default 1000.0).
    kelly_fraction : float
        Fractional Kelly multiplier, 0 < kelly_fraction <= 1 (default 0.25).
    min_edge : float
        Minimum edge required to place a bet (default 0.05).

    Returns
    -------
    dict with keys: roi, final_bankroll, max_drawdown, sharpe_ratio,
    win_rate, n_bets, avg_edge, avg_kelly, history (DataFrame), summary (str).
    """
    if odds_df.empty or model_probs.empty:
        logger.warning("run_backtest: empty input; returning zeroed result.")
        return _empty_result(bankroll)

    # Merge predictions with odds
    data = merge_predictions_with_odds(model_probs, odds_df)
    if data.empty:
        logger.warning("run_backtest: merge produced no rows; returning zeroed result.")
        return _empty_result(bankroll)

    # Sort chronologically so bankroll evolves in time order
    if "date" in data.columns:
        data = data.sort_values("date").reset_index(drop=True)

    outcome_map = {
        "H": ("home_win",  "home_odds"),
        "D": ("draw",      "draw_odds"),
        "A": ("away_win",  "away_odds"),
    }

    current_bankroll = float(bankroll)
    history_rows: list[dict] = []

    for _, row in data.iterrows():
        try:
            actual_outcome = str(row["outcome"]).strip().upper()
            home_odds = float(row["home_odds"])
            draw_odds = float(row["draw_odds"])
            away_odds = float(row["away_odds"])
            p_home = float(row["home_win"])
            p_draw = float(row["draw"])
            p_away = float(row["away_win"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("run_backtest: skipping row — %s", exc)
            continue

        # Skip rows with invalid odds
        if any(o <= 1.0 for o in (home_odds, draw_odds, away_odds)):
            logger.debug("run_backtest: skipping row with odds <= 1.0.")
            continue

        # Fair probabilities (margin removed)
        fair_h, fair_d, fair_a = _remove_margin(home_odds, draw_odds, away_odds)

        edges = {
            "H": (p_home - fair_h, home_odds, p_home),
            "D": (p_draw - fair_d, draw_odds, p_draw),
            "A": (p_away - fair_a, away_odds, p_away),
        }

        # Filter to outcomes with positive edge above threshold
        candidates = {k: v for k, v in edges.items() if v[0] > min_edge}
        if not candidates:
            continue

        # Bet on the single outcome with the highest edge
        best_outcome = max(candidates, key=lambda k: candidates[k][0])
        best_edge, best_odds, best_p = candidates[best_outcome]

        # Kelly fraction (b = decimal_odds - 1)
        b = best_odds - 1.0
        full_kelly = _kelly(best_p, b)
        frac_kelly = kelly_fraction * full_kelly

        bet_size = frac_kelly * current_bankroll
        if bet_size <= 0:
            continue

        won = (actual_outcome == best_outcome)
        pnl = bet_size * b if won else -bet_size
        current_bankroll += pnl
        cumulative_roi = (current_bankroll - bankroll) / bankroll

        history_rows.append({
            "date":           row.get("date", None),
            "home_team":      row.get("home_team", ""),
            "away_team":      row.get("away_team", ""),
            "outcome":        actual_outcome,
            "bet_on":         best_outcome,
            "edge":           round(best_edge, 4),
            "kelly_fraction": round(frac_kelly, 4),
            "bet_size":       round(bet_size, 4),
            "pnl":            round(pnl, 4),
            "bankroll":       round(current_bankroll, 4),
            "cumulative_roi": round(cumulative_roi, 4),
        })

    if not history_rows:
        logger.info("run_backtest: no bets placed (no edges exceeded min_edge=%.3f).", min_edge)
        return _empty_result(bankroll)

    history = pd.DataFrame(history_rows)

    n_bets = len(history)
    total_staked = history["bet_size"].sum()
    total_profit = history["pnl"].sum()
    roi = float(total_profit / total_staked) if total_staked > 0 else 0.0
    win_rate = float((history["pnl"] > 0).mean())
    max_dd = _max_drawdown(history["bankroll"])
    sharpe = _sharpe(history["pnl"])
    avg_edge = float(history["edge"].mean())
    avg_kf = float(history["kelly_fraction"].mean())
    final_bankroll = float(history["bankroll"].iloc[-1])

    direction = "profit" if roi >= 0 else "loss"
    summary = (
        f"{n_bets} bets placed  |  ROI {roi:+.1%}  |  "
        f"final bankroll {final_bankroll:.2f}  |  "
        f"max drawdown {max_dd:.1f}%  |  Sharpe {sharpe:.2f}  "
        f"({direction})"
    )

    logger.info("run_backtest: %s", summary)

    return {
        "roi":            round(roi, 4),
        "final_bankroll": round(final_bankroll, 4),
        "max_drawdown":   round(max_dd, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "win_rate":       round(win_rate, 4),
        "n_bets":         n_bets,
        "avg_edge":       round(avg_edge, 4),
        "avg_kelly":      round(avg_kf, 4),
        "history":        history,
        "summary":        summary,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def backtest_report(result: dict) -> pd.DataFrame:
    """Return a single-row DataFrame of performance metrics.

    Useful for pretty-printing or embedding in a Dash DataTable.

    Parameters
    ----------
    result : dict  output of run_backtest()
    """
    metric_keys = [
        "roi", "final_bankroll", "max_drawdown", "sharpe_ratio",
        "win_rate", "n_bets", "avg_edge", "avg_kelly",
    ]
    return pd.DataFrame(
        [{k: result.get(k, None) for k in metric_keys}]
    )


# ---------------------------------------------------------------------------
# Plotly figures (Dash-compatible)
# ---------------------------------------------------------------------------


def plot_bankroll_history(result: dict) -> "go.Figure":
    """Line chart of bankroll over time with drawdown shading.

    The figure is returned as a go.Figure and can be used directly in a
    Dash layout as:  dcc.Graph(figure=plot_bankroll_history(result))

    Parameters
    ----------
    result : dict  output of run_backtest()
    """
    _require_plotly()

    history = result.get("history", pd.DataFrame())
    roi = result.get("roi", 0.0)
    sharpe = result.get("sharpe_ratio", 0.0)
    initial = result.get("final_bankroll", 1000.0) - history["pnl"].sum() if not history.empty else 1000.0

    fig = go.Figure()

    if history.empty:
        fig.update_layout(
            title="Bankroll history (no bets placed)",
            template="plotly_white",
            height=400,
        )
        return fig

    dates = history["date"].astype(str)
    bankroll = history["bankroll"]

    # Identify drawdown periods (bankroll below running peak)
    peak = bankroll.cummax()
    in_drawdown = bankroll < peak

    # Add shaded drawdown regions as filled areas
    if in_drawdown.any():
        fig.add_trace(
            go.Scatter(
                x=pd.concat([dates, dates[::-1]]),
                y=pd.concat([
                    peak.where(in_drawdown, bankroll),
                    bankroll.where(in_drawdown, bankroll)[::-1],
                ]),
                fill="toself",
                fillcolor="rgba(214, 39, 40, 0.15)",
                line=dict(width=0),
                name="Drawdown",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Initial bankroll reference line
    fig.add_hline(
        y=initial,
        line_dash="dash",
        line_color="#aec7e8",
        annotation_text=f"Initial: {initial:,.0f}",
        annotation_position="bottom right",
    )

    # Bankroll line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bankroll,
            mode="lines+markers",
            name="Bankroll",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Bankroll: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Bankroll history  ·  ROI {roi:+.1%}  ·  Sharpe {sharpe:.2f}"
            ),
            font=dict(size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Bankroll",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    return fig


def plot_edge_distribution(result: dict) -> "go.Figure":
    """Histogram of edges on placed bets, coloured by win / loss.

    Parameters
    ----------
    result : dict  output of run_backtest()
    """
    _require_plotly()

    history = result.get("history", pd.DataFrame())

    fig = go.Figure()

    if history.empty:
        fig.update_layout(
            title="Edge distribution (no bets placed)",
            template="plotly_white",
            height=380,
        )
        return fig

    won_edges = history.loc[history["pnl"] > 0, "edge"]
    lost_edges = history.loc[history["pnl"] <= 0, "edge"]
    mean_edge = float(history["edge"].mean())

    if not won_edges.empty:
        fig.add_trace(
            go.Histogram(
                x=won_edges,
                name="Won",
                marker_color="#2ca02c",
                opacity=0.7,
                nbinsx=20,
                histnorm="",
            )
        )

    if not lost_edges.empty:
        fig.add_trace(
            go.Histogram(
                x=lost_edges,
                name="Lost",
                marker_color="#d62728",
                opacity=0.7,
                nbinsx=20,
                histnorm="",
            )
        )

    fig.add_vline(
        x=mean_edge,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"Mean edge: {mean_edge:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=f"Edge distribution  ·  {len(history)} bets  ·  mean edge {mean_edge:.3f}",
            font=dict(size=14),
        ),
        xaxis_title="Edge (model prob − fair prob)",
        yaxis_title="Number of bets",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def plot_cumulative_roi(result: dict) -> "go.Figure":
    """Cumulative ROI over time with a zero reference line.

    Parameters
    ----------
    result : dict  output of run_backtest()
    """
    _require_plotly()

    history = result.get("history", pd.DataFrame())
    final_roi = result.get("roi", 0.0)

    fig = go.Figure()

    if history.empty:
        fig.update_layout(
            title="Cumulative ROI (no bets placed)",
            template="plotly_white",
            height=380,
        )
        return fig

    dates = history["date"].astype(str)
    cum_roi = history["cumulative_roi"] * 100  # convert to %

    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#aec7e8",
        annotation_text="Break-even",
        annotation_position="bottom right",
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum_roi,
            mode="lines",
            name="Cumulative ROI",
            line=dict(
                color="#2ca02c" if final_roi >= 0 else "#d62728",
                width=2,
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(44, 160, 44, 0.15)" if final_roi >= 0
                else "rgba(214, 39, 40, 0.15)"
            ),
            hovertemplate="Date: %{x}<br>Cumulative ROI: %{y:.2f}%<extra></extra>",
        )
    )

    # Annotate the final ROI value
    if not history.empty:
        fig.add_annotation(
            x=dates.iloc[-1],
            y=float(cum_roi.iloc[-1]),
            text=f"Final ROI: {final_roi:+.1%}",
            showarrow=True,
            arrowhead=2,
            ax=-60,
            ay=-30,
            font=dict(size=11),
        )

    fig.update_layout(
        title=dict(
            text=f"Cumulative ROI  ·  {len(history)} bets  ·  final {final_roi:+.1%}",
            font=dict(size=14),
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative ROI (%)",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    return fig
