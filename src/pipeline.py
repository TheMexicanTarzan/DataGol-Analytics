"""
DataGol pipeline — orchestrates data fetching, personality clustering,
quality scoring, and two-layer matchup prediction.

Two-layer model
---------------
  Layer 1 (personality matchup):
      Neural network predicts xG purely from lineup personality categories.
      This is what the original DataGol notebook trained.

  Layer 2 (quality adjustment):
      Scales the Layer-1 prediction by a quality ratio derived from
      Transfermarkt values + FBref percentiles + Sofascore live ratings.

      home_xg_adjusted = home_xg_raw * quality_multiplier(home, away)
      away_xg_adjusted = away_xg_raw * quality_multiplier(away, home)

      The multiplier uses tanh so extreme quality gaps produce large but
      bounded adjustments. At equal quality both multipliers equal 1.0.

Quick start (notebook cell)
---------------------------
    from src.pipeline import load_tournament_data, load_quality_scores, predict_matchup

    model_df, lineups_df = load_tournament_data("world_cup_2026")
    quality_df           = load_quality_scores("world_cup_2026")

    result = predict_matchup(
        model        = trained_model,       # keras model from notebook
        home_lineup  = [3, 7, 7, 2, 15, 15, 9, 12, 20, 21],   # personality IDs
        away_lineup  = [4, 6, 6, 1, 14, 16, 8, 11, 19, 22],
        home_players = ["Messi", "De Paul", ...],               # for quality lookup
        away_players = ["Mbappé", "Griezmann", ...],
        quality_df   = quality_df,
    )
    print(result)
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    API_FOOTBALL_COMPETITION_MAP,
    APIFootballScraper,
    COMPETITION_REGISTRY,
    DataCache,
    FBrefScraper,
    MultiTournamentLoader,
    PersonalityFeatureBuilder,
    PlayerRegistry,
    QualityScoreBuilder,
    SofascoreScraper,
    TransfermarktScraper,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Validation is imported lazily to keep the module importable without plotly.
def run_validation(
    model_factory,
    X: "np.ndarray",
    y: "np.ndarray",
    dates: "np.ndarray",
    min_train_samples: int = 50,
    test_window_weeks: int = 2,
    fit_kwargs: dict | None = None,
    plot: bool = True,
) -> "tuple[pd.DataFrame, dict]":
    """
    Convenience wrapper: runs walk-forward validation + calibration in one call.

    Parameters
    ----------
    model_factory   : callable → fresh unfitted model
    X, y, dates     : output of load_multi_tournament_data()
    min_train_samples: minimum samples to start a training window
    test_window_weeks: width of each test window
    fit_kwargs      : forwarded to model.fit()  e.g. {'epochs':60,'verbose':0}
    plot            : if True, attach Plotly figures to the returned dict

    Returns
    -------
    (results_df, report) where:
        results_df — per-window walk-forward metrics (pd.DataFrame)
        report     — dict with keys: calibration, figures (if plot=True),
                     overall_mae, overall_baseline_mae, beats_baseline_pct
    """
    from .validation import (
        walk_forward_validate,
        calibration_report,
        compare_vs_baseline,
        plot_walk_forward_results,
        plot_calibration,
        plot_prediction_error_distribution,
    )

    results_df = walk_forward_validate(
        model_factory=model_factory,
        X=X, y=y, dates=dates,
        min_train_samples=min_train_samples,
        test_window_weeks=test_window_weeks,
        fit_kwargs=fit_kwargs or {},
    )

    # Aggregate calibration across ALL held-out windows
    report: dict = {}
    if not results_df.empty:
        report["beats_baseline_pct"] = round(100 * results_df["beats_baseline"].mean(), 1)
        report["overall_mae"] = round(results_df["mae"].mean(), 4)
        report["overall_baseline_mae"] = round(results_df["baseline_mae"].mean(), 4)
        report["summary"] = (
            f"Model beats baseline in {results_df['beats_baseline'].sum()}/"
            f"{len(results_df)} windows "
            f"({report['beats_baseline_pct']}%)  |  "
            f"Mean MAE: {report['overall_mae']} vs baseline {report['overall_baseline_mae']}"
        )
        logger.info(report["summary"])

        if plot:
            report["figures"] = {
                "walk_forward": plot_walk_forward_results(results_df),
            }

    return results_df, report


def register_player_categories(
    clusts_total: pd.DataFrame,
    competition: str,
    db_path: Path = Path("data/player_registry.db"),
) -> PlayerRegistry:
    """
    Persist clustering results from the notebook into the PlayerRegistry.

    Call this once after running the clustering cells in DataGol.ipynb so that
    future runs can resolve player → category without re-clustering.

    Parameters
    ----------
    clusts_total : pd.DataFrame
        Combined cluster assignments with columns: player_name (or player),
        category, position_group.  This is the DataFrame produced after
        concatenating clust_back_df, clust_midfield_df, clust_forward_df.
    competition  : str  e.g. 'copa_america_2024'
    db_path      : Path  where to store the SQLite registry

    Returns
    -------
    PlayerRegistry  (populated and ready to use)
    """
    registry = PlayerRegistry(db_path=db_path)

    df = clusts_total.copy()
    name_col = next((c for c in df.columns if "player" in c.lower() and "id" not in c.lower()), None)
    if name_col and name_col != "player_name":
        df = df.rename(columns={name_col: "player_name"})

    if "position_group" not in df.columns:
        # Infer from category ranges used in the notebook (1-7=def, 8-17=mid, 18-23=fwd, 24+=GK)
        def _infer(cat: int) -> str:
            if cat <= 7:
                return "defender"
            if cat <= 17:
                return "midfielder"
            if cat <= 23:
                return "forward"
            return "goalkeeper"
        df["position_group"] = df["category"].apply(_infer)

    registry.register_bulk(df[["player_name", "category", "position_group"]], competition)
    logger.info("Registered %d players from %s into registry.", len(df), competition)
    return registry


def load_multi_tournament_data(
    competitions: list[str] | None = None,
    international_only: bool = False,
    cache_dir: Path = Path("data/cache"),
    db_path: Path = Path("data/player_registry.db"),
    request_delay: float = 4.0,
) -> tuple[pd.DataFrame, "np.ndarray", "np.ndarray"]:
    """
    Fetch and combine data from multiple competitions for model retraining.

    Replaces the single-tournament ``load_tournament_data`` when you want to
    expand the training set beyond 248 Copa América samples.

    Parameters
    ----------
    competitions     : list of competition keys; None = all in COMPETITION_REGISTRY
    international_only : if True, skip club leagues (use for personality clustering)
    cache_dir        : parquet cache (historical comps use ttl_hours=720)
    db_path          : PlayerRegistry SQLite path (must be pre-populated via
                       ``register_player_categories`` after the clustering notebook runs)
    request_delay    : seconds between HTTP requests (be polite to FBref)

    Returns
    -------
    player_features : pd.DataFrame
        Combined feature matrix (all players × personality features), ready for
        re-clustering.  Includes `competition` and `sample_weight` columns.
    X : np.ndarray, shape (N, 2, 11)
        Lineup arrays for training.  Axis 2 = [GK, 10 outfield], values = category IDs.
    y : np.ndarray, shape (N,)
        Goals scored by the home team per lineup snapshot.
    """
    registry = PlayerRegistry(db_path=db_path)
    loader = MultiTournamentLoader(
        competitions=competitions,
        cache_dir=cache_dir,
        ttl_hours=720,
        request_delay=request_delay,
        international_only=international_only,
    )

    logger.info("Loading player features from multiple competitions …")
    player_features = loader.load_player_features()

    logger.info("Loading training pairs (lineups + goals) …")
    X, y, dates = loader.load_training_pairs(registry)

    logger.info(
        "Multi-tournament load complete: %d players, %d lineup samples.",
        len(player_features), len(X),
    )
    return player_features, X, y, dates


def load_tournament_data(
    competition: str = "world_cup_2026",
    cache_dir: Path = Path("data/cache"),
    ttl_hours: int = 6,
    min_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch, clean, and featurise all player data for the given tournament.

    Source priority:
      1. API-Football   — when API_FOOTBALL_KEY env var is set and the
                          competition is in API_FOOTBALL_COMPETITION_MAP.
      2. FBref/StatsBomb — fallback (StatsBomb open data for supported
                           tournaments, direct FBref scraping otherwise).

    Returns
    -------
    model_df : pd.DataFrame
        Feature matrix (players × personality features), ready for clustering.
    lineups_df : pd.DataFrame
        Long-format lineups: [fixture_id/match_url, side, player, ...].
    """
    _load_env()

    builder = PersonalityFeatureBuilder()
    builder.MIN_MINUTES = min_minutes

    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    if api_key and competition in API_FOOTBALL_COMPETITION_MAP:
        logger.info("API-Football: fetching data for '%s' …", competition)
        af = APIFootballScraper(cache_dir=cache_dir, ttl_hours=ttl_hours)
        try:
            merged_stats = af.get_merged_player_stats(competition)
            model_df     = builder.build(merged_stats)

            schedule     = af.get_schedule(competition)
            fixture_ids  = (
                schedule["fixture_id"].dropna().astype(int).tolist()
                if not schedule.empty else []
            )
            lineups_df = af.get_all_lineups(fixture_ids)

            logger.info(
                "API-Football pipeline complete: %d players, %d lineup records.",
                len(model_df), len(lineups_df),
            )
            return model_df, lineups_df
        except Exception as exc:
            logger.warning("API-Football failed (%s) — falling back to FBref/StatsBomb.", exc)

    # Fallback: FBref (with StatsBomb open data for supported competitions)
    scraper = FBrefScraper(competition=competition, cache_dir=cache_dir, ttl_hours=ttl_hours)

    logger.info("Fetching player stats for '%s' …", competition)
    merged_stats = scraper.get_merged_player_stats()

    logger.info("Building personality features …")
    model_df = builder.build(merged_stats)

    logger.info("Fetching match schedule …")
    schedule = scraper.get_schedule()

    logger.info("Fetching match lineups …")
    match_urls = _extract_match_urls(schedule)
    lineups_df = scraper.get_all_lineups(match_urls)

    logger.info("Pipeline complete: %d players, %d lineup records.", len(model_df), len(lineups_df))
    return model_df, lineups_df


def load_quality_scores(
    competition: str = "world_cup_2026",
    fbref_stats: pd.DataFrame | None = None,
    cache_dir: Path = Path("data/cache"),
    ttl_hours: int = 6,
    w_transfermarkt: float = 0.40,
    w_fbref: float = 0.30,
    w_sofascore: float = 0.30,
) -> pd.DataFrame:
    """
    Fetch and combine quality scores from Transfermarkt, FBref, and Sofascore.

    Parameters
    ----------
    competition   : tournament key (e.g. 'world_cup_2026')
    fbref_stats   : pre-fetched merged FBref stats; re-fetched if None
    cache_dir     : parquet cache directory
    ttl_hours     : cache TTL (use 6 during the tournament)
    w_transfermarkt / w_fbref / w_sofascore : source weights (must sum to 1)

    Returns
    -------
    DataFrame indexed by player name with columns:
      quality_score, tm_score, fbref_score, sofascore_score,
      market_value_eur, sofascore_rating, appearances, position_group
    """
    cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
    cache_key = f"{competition}_quality_scores"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # FBref stats (reuse if already loaded)
    if fbref_stats is None:
        scraper = FBrefScraper(competition=competition, cache_dir=cache_dir, ttl_hours=ttl_hours)
        fbref_stats = scraper.get_merged_player_stats()

    # Transfermarkt
    tm_values: pd.DataFrame | None = None
    try:
        logger.info("Fetching Transfermarkt values for '%s' …", competition)
        tm_scraper = TransfermarktScraper()
        tm_values = tm_scraper.get_player_values(competition)
        logger.info("Transfermarkt: %d players", len(tm_values))
    except Exception as exc:
        logger.warning("Transfermarkt fetch failed (continuing without it): %s", exc)

    # Sofascore
    sofascore_ratings: pd.DataFrame | None = None
    try:
        logger.info("Fetching Sofascore ratings for '%s' …", competition)
        sof_scraper = SofascoreScraper()
        sofascore_ratings = sof_scraper.get_player_ratings(competition)
        logger.info("Sofascore: %d players", len(sofascore_ratings))
    except Exception as exc:
        logger.warning("Sofascore fetch failed (continuing without it): %s", exc)

    builder = QualityScoreBuilder(
        w_transfermarkt=w_transfermarkt,
        w_fbref=w_fbref,
        w_sofascore=w_sofascore,
    )
    quality_df = builder.build(
        fbref_stats=fbref_stats,
        tm_values=tm_values,
        sofascore_ratings=sofascore_ratings,
    )

    cache.set(cache_key, quality_df.reset_index())
    logger.info("Quality scores built for %d players.", len(quality_df))
    return quality_df


# ---------------------------------------------------------------------------
# Two-layer matchup prediction
# ---------------------------------------------------------------------------


def predict_matchup(
    model,
    home_lineup: list[int],
    away_lineup: list[int],
    home_players: list[str] | None = None,
    away_players: list[str] | None = None,
    quality_df: pd.DataFrame | None = None,
    quality_sensitivity: float = 1.5,
) -> dict:
    """
    Predict expected goals for a head-to-head matchup.

    Layer 1 uses the trained neural network (personality matchup only).
    Layer 2 adjusts for team quality when quality_df and player names are given.

    Parameters
    ----------
    model            : trained Keras model (from DataGol notebook)
    home_lineup      : list of 10 personality category IDs for the home team
    away_lineup      : list of 10 personality category IDs for the away team
    home_players     : player names in same order as home_lineup (optional)
    away_players     : player names in same order as away_lineup (optional)
    quality_df       : output of load_quality_scores()  (optional)
    quality_sensitivity : tanh steepness; higher = larger quality effect

    Returns
    -------
    dict with keys:
      raw_home_xg        — Layer-1 prediction (personality only)
      raw_away_xg
      adjusted_home_xg   — Layer-2 prediction (personality + quality)
      adjusted_away_xg
      quality_adjustment — multiplier applied to home team (>1 = home favoured)
      home_win_prob      — P(home goals > away goals) from adjusted predictions
      expected_goal_diff — adjusted_home_xg - adjusted_away_xg
    """
    # Layer 1: personality matchup
    home_arr = np.array(home_lineup, dtype=float).reshape(1, 1, 10)
    away_arr = np.array(away_lineup, dtype=float).reshape(1, 1, 10)
    lineup_input = np.concatenate([home_arr, away_arr], axis=1)  # (1, 2, 10)

    raw_home_xg = float(model.predict(lineup_input, verbose=0)[0][0])

    away_input = np.concatenate([away_arr, home_arr], axis=1)
    raw_away_xg = float(model.predict(away_input, verbose=0)[0][0])

    # Layer 2: quality adjustment
    adj = 1.0
    if quality_df is not None and home_players and away_players:
        adj = _quality_multiplier(
            home_players, away_players, quality_df, quality_sensitivity
        )

    adjusted_home_xg = raw_home_xg * adj
    adjusted_away_xg = raw_away_xg * (2.0 - adj)  # symmetric: equal adj pushes to 1.0

    # Win probability (Poisson approximation)
    home_win_p = _poisson_win_prob(adjusted_home_xg, adjusted_away_xg)
    away_win_p = _poisson_win_prob(adjusted_away_xg, adjusted_home_xg)
    draw_p = max(0.0, 1.0 - home_win_p - away_win_p)

    return {
        "raw_home_xg": round(raw_home_xg, 3),
        "raw_away_xg": round(raw_away_xg, 3),
        "adjusted_home_xg": round(adjusted_home_xg, 3),
        "adjusted_away_xg": round(adjusted_away_xg, 3),
        "quality_adjustment": round(adj, 4),
        "home_win_prob": round(home_win_p, 3),
        "draw_prob": round(draw_p, 3),
        "away_win_prob": round(away_win_p, 3),
        "expected_goal_diff": round(adjusted_home_xg - adjusted_away_xg, 3),
    }


# ---------------------------------------------------------------------------
# Phase 3.2 — Confidence intervals
# ---------------------------------------------------------------------------


def predict_with_confidence(
    model,
    home_lineup: list[int],
    away_lineup: list[int],
    home_players: list[str] | None = None,
    away_players: list[str] | None = None,
    quality_df: pd.DataFrame | None = None,
    quality_sensitivity: float = 1.5,
    mae_estimate: float | None = None,
    model_ensemble: list | None = None,
    confidence_level: float = 0.90,
) -> dict:
    """
    Like ``predict_matchup`` but adds prediction intervals to every xG and
    probability field.

    Interval strategy (choose one):

    A) ``model_ensemble`` provided — run each model, take percentiles.
       Most accurate; requires N trained models (e.g. from different training
       folds or bootstrap samples).

    B) ``mae_estimate`` provided — Gaussian approximation around the point
       estimate using the MAE from walk-forward validation as the scale.
       Faster; use ``run_validation()`` results to get a realistic MAE.

    C) Neither provided — returns the plain ``predict_matchup`` result with
       ``*_low`` / ``*_high`` fields set to the point estimate (no interval).

    Parameters
    ----------
    model           : primary trained model (used for the point estimate)
    mae_estimate    : float, optional — MAE from walk-forward validation;
                      90% CI half-width ≈ 1.645 * MAE
    model_ensemble  : list of trained models, optional — used for interval
                      estimation; overrides mae_estimate if both are provided
    confidence_level: float in (0,1), default 0.90 — CI coverage

    Returns
    -------
    All keys from ``predict_matchup`` plus:
        adjusted_home_xg_low / _high
        adjusted_away_xg_low / _high
        home_win_prob_low / _high
        away_win_prob_low / _high
        draw_prob_low / _high
        ci_method  : 'ensemble' | 'gaussian' | 'none'
    """
    base = predict_matchup(
        model, home_lineup, away_lineup,
        home_players, away_players, quality_df, quality_sensitivity,
    )

    alpha = (1.0 - confidence_level) / 2.0  # tail probability on each side
    h_xg = base["adjusted_home_xg"]
    a_xg = base["adjusted_away_xg"]

    if model_ensemble:
        h_preds, a_preds = [], []
        for m in model_ensemble:
            r = predict_matchup(m, home_lineup, away_lineup,
                                home_players, away_players, quality_df, quality_sensitivity)
            h_preds.append(r["adjusted_home_xg"])
            a_preds.append(r["adjusted_away_xg"])
        h_low  = float(np.quantile(h_preds, alpha))
        h_high = float(np.quantile(h_preds, 1 - alpha))
        a_low  = float(np.quantile(a_preds, alpha))
        a_high = float(np.quantile(a_preds, 1 - alpha))
        ci_method = "ensemble"

    elif mae_estimate is not None:
        from scipy.stats import norm
        z = norm.ppf(1 - alpha)  # e.g. 1.645 for 90%
        half = z * float(mae_estimate)
        h_low, h_high = max(0.0, h_xg - half), h_xg + half
        a_low, a_high = max(0.0, a_xg - half), a_xg + half
        ci_method = "gaussian"

    else:
        h_low = h_high = h_xg
        a_low = a_high = a_xg
        ci_method = "none"

    def _probs(hxg, axg):
        hp = _poisson_win_prob(hxg, axg)
        ap = _poisson_win_prob(axg, hxg)
        return round(hp, 3), round(max(0.0, 1 - hp - ap), 3), round(ap, 3)

    hw_lo, dr_lo, aw_lo = _probs(h_low,  a_high)  # pessimistic for home
    hw_hi, dr_hi, aw_hi = _probs(h_high, a_low)   # optimistic for home

    return {
        **base,
        "adjusted_home_xg_low":  round(h_low,  3),
        "adjusted_home_xg_high": round(h_high, 3),
        "adjusted_away_xg_low":  round(a_low,  3),
        "adjusted_away_xg_high": round(a_high, 3),
        "home_win_prob_low":  min(hw_lo, hw_hi),
        "home_win_prob_high": max(hw_lo, hw_hi),
        "draw_prob_low":  min(dr_lo, dr_hi),
        "draw_prob_high": max(dr_lo, dr_hi),
        "away_win_prob_low":  min(aw_lo, aw_hi),
        "away_win_prob_high": max(aw_lo, aw_hi),
        "ci_method":   ci_method,
        "confidence_level": confidence_level,
    }


# ---------------------------------------------------------------------------
# Phase 3.3 — Betting signal + Kelly Criterion
# ---------------------------------------------------------------------------


def betting_signal(
    matchup_result: dict,
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
) -> dict:
    """
    Translate a matchup prediction into actionable betting signals.

    The bookmaker margin is removed before computing edges so the comparison
    is fair: model probability vs the true implied probability (not the raw
    inverse of the odds).

    Parameters
    ----------
    matchup_result  : dict — output of predict_matchup() or
                      predict_with_confidence()
    home_odds       : decimal odds for home win  (e.g. 2.50)
    draw_odds       : decimal odds for draw      (e.g. 3.20)
    away_odds       : decimal odds for away win  (e.g 2.80)
    kelly_fraction  : fractional Kelly multiplier (0.25 = quarter-Kelly)
    min_edge        : minimum edge to emit a signal; filters noise

    Returns
    -------
    dict with keys:
      bookmaker_margin  : vig as a fraction (e.g. 0.05 = 5%)
      signals           : list[dict] — one per outcome with positive edge,
                          sorted by edge descending. Each signal has:
                            outcome         : 'home' | 'draw' | 'away'
                            model_prob      : float
                            fair_prob       : float  (margin-removed)
                            decimal_odds    : float
                            edge            : model_prob - fair_prob
                            full_kelly      : float
                            bet_fraction    : full_kelly * kelly_fraction
                            expected_value  : edge * decimal_odds
                            recommendation  : str  (human-readable)
      best_signal       : dict | None  — highest-edge signal, or None
      value_bet_found   : bool
    """
    raw_probs = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    overround = sum(raw_probs)
    margin = round(overround - 1.0, 4)
    fair_probs = [p / overround for p in raw_probs]

    model_probs = [
        matchup_result.get("home_win_prob", 0.0),
        matchup_result.get("draw_prob", 0.0),
        matchup_result.get("away_win_prob", 0.0),
    ]
    odds_list   = [home_odds, draw_odds, away_odds]
    labels      = ["home", "draw", "away"]

    signals = []
    for label, mp, fp, dec_odds in zip(labels, model_probs, fair_probs, odds_list):
        edge = round(mp - fp, 4)
        if edge <= min_edge:
            continue
        b = dec_odds - 1.0
        full_kelly = round(max(0.0, (b * mp - (1 - mp)) / b), 4)
        bet_frac   = round(full_kelly * kelly_fraction, 4)
        ev         = round(edge * dec_odds, 4)

        if bet_frac > 0:
            rec = (
                f"BET {label.upper()}: stake {bet_frac*100:.1f}% of bankroll  "
                f"(edge={edge:+.1%}, EV={ev:+.3f} per unit, "
                f"fair odds={1/fp:.2f} vs market {dec_odds:.2f})"
            )
        else:
            rec = f"Edge found ({edge:+.1%}) but Kelly=0 — skip."

        signals.append({
            "outcome":        label,
            "model_prob":     round(mp,  4),
            "fair_prob":      round(fp,  4),
            "decimal_odds":   dec_odds,
            "edge":           edge,
            "full_kelly":     full_kelly,
            "bet_fraction":   bet_frac,
            "expected_value": ev,
            "recommendation": rec,
        })

    signals.sort(key=lambda s: s["edge"], reverse=True)
    best = signals[0] if signals else None

    if best:
        logger.info("Betting signal: %s", best["recommendation"])

    return {
        "bookmaker_margin": margin,
        "signals":          signals,
        "best_signal":      best,
        "value_bet_found":  bool(signals),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quality_multiplier(
    home_players: list[str],
    away_players: list[str],
    quality_df: pd.DataFrame,
    sensitivity: float,
) -> float:
    """
    Compute the quality adjustment multiplier for the home team.

    multiplier = 1 + tanh(sensitivity * (q_home - q_away) / (q_home + q_away))

    At equal quality: multiplier = 1.0  (no adjustment)
    Home much stronger: multiplier → 2.0 (double xG)
    Home much weaker:   multiplier → 0.0 (halved xG)
    """
    idx = quality_df.index.tolist()

    def avg_quality(players: list[str]) -> float:
        scores = []
        for name in players:
            from difflib import get_close_matches
            import unicodedata

            def _norm(s):
                nfkd = unicodedata.normalize("NFKD", str(s))
                return " ".join(nfkd.encode("ascii", "ignore").decode().lower().split())

            matches = get_close_matches(_norm(name), [_norm(i) for i in idx], n=1, cutoff=0.75)
            if matches:
                orig = idx[[_norm(i) for i in idx].index(matches[0])]
                scores.append(quality_df.loc[orig, "quality_score"])
        return float(np.mean(scores)) if scores else 0.5

    q_home = avg_quality(home_players)
    q_away = avg_quality(away_players)

    denom = q_home + q_away
    if denom < 1e-8:
        return 1.0

    ratio = (q_home - q_away) / denom
    multiplier = 1.0 + float(np.tanh(sensitivity * ratio))
    return float(np.clip(multiplier, 0.1, 1.9))  # safety bounds


def _poisson_win_prob(lambda_a: float, lambda_b: float, max_goals: int = 10) -> float:
    """P(Poisson(lambda_a) > Poisson(lambda_b)) via exact sum."""
    from math import exp, factorial

    def pmf(lam: float, k: int) -> float:
        return exp(-lam) * (lam ** k) / factorial(k)

    prob = 0.0
    for a in range(1, max_goals + 1):
        for b in range(0, a):
            prob += pmf(lambda_a, a) * pmf(lambda_b, b)
    return float(np.clip(prob, 0.0, 1.0))


def _load_env() -> None:
    """Load .env into os.environ if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _extract_match_urls(schedule: pd.DataFrame) -> list[str]:
    url_col = next(
        (c for c in schedule.columns if "url" in c.lower() or "report" in c.lower()), None
    )
    if url_col is None:
        logger.warning("No match URL column found in schedule. Lineup scraping skipped.")
        return []
    base = "https://fbref.com"
    urls = schedule[url_col].dropna().tolist()
    return [u if u.startswith("http") else f"{base}{u}" for u in urls]
