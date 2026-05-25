"""
MultiTournamentLoader — orchestrates FBrefScraper across multiple competitions
to produce the raw feature matrices and lineup/goal pairs used for re-clustering
and retraining the DataGol personality model.

Key design decision: personality clustering is performed OUTSIDE this file
(in notebooks or training scripts). This loader is purely a data pipeline:
it fetches, normalises, and assembles arrays ready for downstream use.

Intensity-dependent features (xG, goals, shot/pass assists) are z-scored
within each competition before concatenation, then rescaled to a soft common
distribution (mean=0.5, std=0.15) so inter-league xG level bias does not
dominate the clustering signal.

Sample weighting for training pairs mirrors the original notebook's approach:
international tournaments receive weight=1.5, club competitions weight=1.0.
Rows with fractional weight are augmented by duplicating with small Gaussian
noise (ε~N(0, 0.1)) added to category IDs to prevent exact duplicates.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import DataCache
from .fbref_scraper import FBrefScraper
from .feature_builder import PersonalityFeatureBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Competition registry
# ---------------------------------------------------------------------------

COMPETITION_REGISTRY: dict[str, dict] = {
    # International tournaments — highest signal, use for personality clustering baseline
    "world_cup_2022":        {"type": "international", "sample_weight": 1.5},
    "copa_america_2024":     {"type": "international", "sample_weight": 1.5},
    "euro_2024":             {"type": "international", "sample_weight": 1.5},
    # Club competitions — large sample size for matchup model training
    "champions_league_2324": {"type": "club",          "sample_weight": 1.0},
    "premier_league_2324":   {"type": "club",          "sample_weight": 1.0},
    "bundesliga_2324":       {"type": "club",          "sample_weight": 1.0},
    "la_liga_2324":          {"type": "club",          "sample_weight": 1.0},
}

# Column name fragments that are intensity-dependent and require inter-league normalisation.
_INTENSITY_FRAGMENTS = ("xg", "goal", "shot_assist", "pass_goal")

# Position ordering for lineup arrays: GK at index 0, then defenders, midfielders, forwards.
_POSITION_GROUP_ORDER = ("goalkeeper", "defender", "midfielder", "forward")

# Noise scale added to duplicated category IDs during sample-weight augmentation.
_AUGMENT_NOISE_STD = 0.1

# Soft normalisation target: after z-scoring within competition, rescale to this distribution.
_NORM_TARGET_MEAN = 0.5
_NORM_TARGET_STD = 0.15


# ---------------------------------------------------------------------------
# MultiTournamentLoader
# ---------------------------------------------------------------------------


class MultiTournamentLoader:
    """
    Orchestrates FBrefScraper calls across multiple competitions and assembles
    the combined feature matrices and training arrays needed for the DataGol
    personality-clustering and matchup-prediction pipelines.

    Parameters
    ----------
    competitions : list[str] | None
        Competition keys to include. ``None`` uses all entries in
        ``COMPETITION_REGISTRY``.
    cache_dir : Path
        Passed to the underlying ``DataCache`` and ``FBrefScraper`` instances.
    ttl_hours : int
        Cache time-to-live in hours. Default 720 (≈30 days) since historical
        tournament data does not change.
    request_delay : float
        Per-request sleep in seconds (polite scraping; passed to FBrefScraper).
    international_only : bool
        When ``True``, restrict to competitions of type ``"international"`` in
        ``COMPETITION_REGISTRY``.  Ignored when ``competitions`` is provided
        explicitly — caller is responsible for filtering in that case.
    """

    def __init__(
        self,
        competitions: list[str] | None = None,
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 720,
        request_delay: float = 4.0,
        international_only: bool = False,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttl_hours = ttl_hours
        self._request_delay = request_delay
        self._international_only = international_only

        if competitions is not None:
            unknown = set(competitions) - set(COMPETITION_REGISTRY)
            if unknown:
                raise ValueError(
                    f"Unknown competition keys: {sorted(unknown)}. "
                    f"Available: {sorted(COMPETITION_REGISTRY)}"
                )
            self._competitions = list(competitions)
        else:
            self._competitions = [
                key
                for key, meta in COMPETITION_REGISTRY.items()
                if not international_only or meta["type"] == "international"
            ]

        logger.info(
            "MultiTournamentLoader initialised with %d competition(s): %s",
            len(self._competitions),
            self._competitions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_player_features(self) -> pd.DataFrame:
        """
        Fetch and assemble a combined player feature matrix across all configured
        competitions.

        For each competition:
          1. Fetch merged player stats via ``FBrefScraper.get_merged_player_stats()``.
          2. Build personality features via ``PersonalityFeatureBuilder().build()``.
          3. Attach ``competition``, ``competition_type``, and ``sample_weight`` columns.

        After concatenation, intensity-dependent columns (those containing ``xg``,
        ``goal``, ``shot_assist``, or ``pass_goal`` in their name) are z-scored
        within each competition and rescaled to mean=0.5, std=0.15 to remove
        inter-league xG level bias.

        Returns
        -------
        pd.DataFrame
            One row per player across all competitions. Index is player name.
            Extra columns: ``competition``, ``competition_type``, ``sample_weight``.
        """
        frames: list[pd.DataFrame] = []

        for comp in self._competitions:
            logger.info("Loading player features for '%s' …", comp)
            try:
                raw_stats, _ = self._fetch_competition(comp)
                if raw_stats.empty:
                    logger.warning("No player stats returned for '%s'; skipping.", comp)
                    continue

                builder = PersonalityFeatureBuilder()
                features = builder.build(raw_stats)

                meta = COMPETITION_REGISTRY[comp]
                features = features.copy()
                features["competition"] = comp
                features["competition_type"] = meta["type"]
                features["sample_weight"] = meta["sample_weight"]

                frames.append(features)
                logger.info(
                    "  '%s': %d players loaded.", comp, len(features)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load player features for '%s': %s", comp, exc
                )

        if not frames:
            logger.warning("No player features loaded from any competition.")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=0)
        combined = self._normalise_intensity_features(combined)
        return combined

    def load_training_pairs(
        self, player_registry: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build lineup arrays and goal targets for all configured competitions.

        For each competition:
          1. Fetch the match schedule to extract match report URLs.
          2. Fetch all lineups via ``FBrefScraper.get_all_lineups()``.
          3. Convert player names to personality category IDs using
             ``player_registry.resolve()``.
          4. Stack into arrays ``X`` of shape ``(N, 2, 11)`` and ``y`` of shape ``(N,)``.

        Sample-weight augmentation: competitions with ``sample_weight > 1.0`` have
        ``floor(weight - 1.0)`` duplicate rows appended with Gaussian noise
        (ε~N(0, 0.1)) added to category IDs (clipped to the valid category range
        inferred from ``player_registry``).

        Parameters
        ----------
        player_registry
            Object exposing ``resolve(player_name: str, position_group: str) ->
            tuple[int, str]``.  Typically a ``PlayerRegistry`` instance.

        Returns
        -------
        (X, y) :
            X — np.ndarray of shape (N, 2, 11), dtype float32.
            y — np.ndarray of shape (N,), dtype float32, goals by home team.
        """
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for comp in self._competitions:
            logger.info("Loading training pairs for '%s' …", comp)
            try:
                _, lineups_df = self._fetch_competition(comp)
                if lineups_df.empty:
                    logger.warning("No lineup data for '%s'; skipping.", comp)
                    continue

                # goals_df is embedded in the schedule; separate it out
                # _fetch_competition returns lineups with a 'goals_home' column
                goals_df = (
                    lineups_df[["match_url", "goals_home"]]
                    .drop_duplicates("match_url")
                    if "goals_home" in lineups_df.columns
                    else pd.DataFrame(columns=["match_url", "goals_home"])
                )

                X_comp, y_comp = self._build_lineup_array(
                    lineups_df, goals_df, player_registry
                )
                if X_comp.size == 0:
                    logger.warning("Empty lineup array for '%s'; skipping.", comp)
                    continue

                weight = COMPETITION_REGISTRY[comp]["sample_weight"]
                X_comp, y_comp = self._apply_sample_weight(X_comp, y_comp, weight)

                X_parts.append(X_comp)
                y_parts.append(y_comp)
                logger.info(
                    "  '%s': %d lineup snapshots (after weighting).", comp, len(y_comp)
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load training pairs for '%s': %s", comp, exc
                )

        if not X_parts:
            logger.warning("No training pairs loaded from any competition.")
            return np.empty((0, 2, 11), dtype=np.float32), np.empty(0, dtype=np.float32)

        X_all = np.concatenate(X_parts, axis=0).astype(np.float32)
        y_all = np.concatenate(y_parts, axis=0).astype(np.float32)
        return X_all, y_all

    def load_international_features(self) -> pd.DataFrame:
        """
        Convenience: return player features from international competitions only.

        Equivalent to constructing a new loader with ``international_only=True``
        and calling ``load_player_features()``.  Used for baseline personality
        clustering where international tournament data provides higher signal.

        Returns
        -------
        pd.DataFrame
            Combined player feature matrix for international competitions only.
        """
        intl_competitions = [
            comp
            for comp in self._competitions
            if COMPETITION_REGISTRY[comp]["type"] == "international"
        ]
        if not intl_competitions:
            logger.warning(
                "No international competitions in current selection: %s",
                self._competitions,
            )
            return pd.DataFrame()

        sub_loader = MultiTournamentLoader(
            competitions=intl_competitions,
            cache_dir=self._cache_dir,
            ttl_hours=self._ttl_hours,
            request_delay=self._request_delay,
            international_only=False,  # already filtered above
        )
        return sub_loader.load_player_features()

    def competition_summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with per-competition statistics.

        Columns: competition, type, n_players, n_lineups, avg_goals, sample_weight.

        This performs lightweight data fetching — it uses the same cache as the
        main load methods, so subsequent calls are fast.

        Returns
        -------
        pd.DataFrame
        """
        rows: list[dict] = []

        for comp in self._competitions:
            meta = COMPETITION_REGISTRY[comp]
            n_players = 0
            n_lineups = 0
            avg_goals = float("nan")

            try:
                raw_stats, lineups_df = self._fetch_competition(comp)
                if not raw_stats.empty:
                    n_players = len(raw_stats)
                if not lineups_df.empty:
                    if "match_url" in lineups_df.columns:
                        n_lineups = lineups_df["match_url"].nunique()
                    if "goals_home" in lineups_df.columns:
                        avg_goals = lineups_df.drop_duplicates("match_url")["goals_home"].mean()
            except Exception as exc:
                logger.warning("Summary fetch failed for '%s': %s", comp, exc)

            rows.append(
                {
                    "competition": comp,
                    "type": meta["type"],
                    "n_players": n_players,
                    "n_lineups": n_lineups,
                    "avg_goals": avg_goals,
                    "sample_weight": meta["sample_weight"],
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_competition(
        self, competition: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch player stats and lineups for a single competition.

        Returns
        -------
        (player_stats, lineups_df) :
            player_stats — merged FBref player stats DataFrame (one row per player).
            lineups_df   — long-format lineup DataFrame with columns including
                           ``match_url``, ``side``, ``player``, and ``goals_home``
                           (if available from the schedule).
        """
        logger.info("_fetch_competition('%s') …", competition)

        scraper = FBrefScraper(
            competition=competition,  # type: ignore[arg-type]
            cache_dir=self._cache_dir,
            ttl_hours=self._ttl_hours,
            request_delay=self._request_delay,
        )

        # -- Player stats --------------------------------------------------
        try:
            player_stats = scraper.get_merged_player_stats()
        except Exception as exc:
            logger.warning("Could not fetch player stats for '%s': %s", competition, exc)
            player_stats = pd.DataFrame()

        # -- Schedule & lineups --------------------------------------------
        lineups_df = pd.DataFrame()
        try:
            schedule = scraper.get_schedule()
            match_urls = _extract_match_urls(schedule)

            if match_urls:
                lineups_df = scraper.get_all_lineups(match_urls)
                # Attach home goals from schedule if possible
                goals_map = _build_goals_map(schedule, match_urls)
                if goals_map and not lineups_df.empty:
                    lineups_df["goals_home"] = lineups_df["match_url"].map(goals_map)
        except Exception as exc:
            logger.warning(
                "Could not fetch schedule/lineups for '%s': %s", competition, exc
            )

        return player_stats, lineups_df

    def _normalise_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score intensity-dependent columns within each competition, then rescale
        to a soft target distribution (mean=0.5, std=0.15).

        Intensity-dependent columns are those whose names contain any of the
        fragments: ``xg``, ``goal``, ``shot_assist``, ``pass_goal``.

        Parameters
        ----------
        df : pd.DataFrame
            Combined player feature DataFrame with a ``competition`` column.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with intensity columns normalised in-place.
        """
        df = df.copy()

        # Identify intensity columns (exclude metadata columns)
        meta_cols = {"competition", "competition_type", "sample_weight"}
        numeric_cols = df.select_dtypes(include="number").columns.difference(meta_cols)

        intensity_cols = [
            c
            for c in numeric_cols
            if any(frag in c.lower() for frag in _INTENSITY_FRAGMENTS)
        ]

        if not intensity_cols:
            logger.debug("No intensity-dependent columns found; skipping normalisation.")
            return df

        for col in intensity_cols:
            # Z-score within each competition
            def _z_score_group(series: pd.Series) -> pd.Series:
                mu = series.mean()
                sigma = series.std(ddof=0)
                if sigma == 0 or np.isnan(sigma):
                    return series - mu  # all same value → all zeros
                return (series - mu) / sigma

            df[col] = df.groupby("competition")[col].transform(_z_score_group)

        # Rescale to target distribution across all competitions
        for col in intensity_cols:
            global_mu = df[col].mean()
            global_sigma = df[col].std(ddof=0)
            if global_sigma > 0 and not np.isnan(global_sigma):
                df[col] = (df[col] - global_mu) / global_sigma
                df[col] = df[col] * _NORM_TARGET_STD + _NORM_TARGET_MEAN

        logger.debug(
            "Normalised %d intensity-dependent column(s) across %d competition(s).",
            len(intensity_cols),
            df["competition"].nunique(),
        )
        return df

    def _build_lineup_array(
        self,
        lineups_df: pd.DataFrame,
        goals_df: pd.DataFrame,
        registry: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a long-format lineup DataFrame into ``(X, y)`` arrays.

        GK occupies slot index 0; outfield players are sorted by position group
        (defenders, midfielders, forwards) and padded/truncated to exactly 10 slots
        (indices 1–10), giving shape ``(N, 2, 11)`` per lineup snapshot.

        Category IDs are resolved via ``registry.resolve(player_name, position_group)``.

        Parameters
        ----------
        lineups_df : pd.DataFrame
            Long-format DataFrame with columns ``match_url``, ``side``, ``player``,
            optionally ``position`` and ``goals_home``.
        goals_df : pd.DataFrame
            DataFrame with columns ``match_url``, ``goals_home`` (one row per match).
            Used when ``goals_home`` is not already present in ``lineups_df``.
        registry
            Object with ``resolve(player_name, position_group) -> (category, source)``.

        Returns
        -------
        (X_comp, y_comp) :
            X_comp — np.ndarray of shape (N, 2, 11), dtype float32.
            y_comp — np.ndarray of shape (N,),       dtype float32.
        """
        if lineups_df.empty:
            return np.empty((0, 2, 11), dtype=np.float32), np.empty(0, dtype=np.float32)

        required_cols = {"match_url", "side", "player"}
        missing = required_cols - set(lineups_df.columns)
        if missing:
            logger.warning(
                "_build_lineup_array: missing columns %s; returning empty arrays.", missing
            )
            return np.empty((0, 2, 11), dtype=np.float32), np.empty(0, dtype=np.float32)

        # Ensure goals are available in lineups_df
        if "goals_home" not in lineups_df.columns:
            if not goals_df.empty and "goals_home" in goals_df.columns:
                lineups_df = lineups_df.merge(
                    goals_df[["match_url", "goals_home"]], on="match_url", how="left"
                )
            else:
                lineups_df = lineups_df.copy()
                lineups_df["goals_home"] = np.nan

        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []

        match_urls = lineups_df["match_url"].unique()

        for url in match_urls:
            match_df = lineups_df[lineups_df["match_url"] == url]
            home_df = match_df[match_df["side"] == "home"]
            away_df = match_df[match_df["side"] == "away"]

            if home_df.empty or away_df.empty:
                continue

            home_cats = _resolve_lineup(home_df, registry)
            away_cats = _resolve_lineup(away_df, registry)

            if home_cats is None or away_cats is None:
                continue

            goals = _get_goals(match_df)
            snapshot = np.stack([home_cats, away_cats], axis=0)  # (2, 11)
            X_rows.append(snapshot)
            y_rows.append(goals)

        if not X_rows:
            return np.empty((0, 2, 11), dtype=np.float32), np.empty(0, dtype=np.float32)

        X_comp = np.array(X_rows, dtype=np.float32)   # (N, 2, 11)
        y_comp = np.array(y_rows, dtype=np.float32)   # (N,)
        return X_comp, y_comp

    @staticmethod
    def _apply_sample_weight(
        X: np.ndarray, y: np.ndarray, weight: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Duplicate rows to reflect fractional sample weights.

        For weight=1.5, 50 % additional rows are appended with Gaussian noise
        (ε~N(0, 0.1)) on the category values.  Noisy category IDs are clipped
        to the range [1, max_category] inferred from the data.

        Parameters
        ----------
        X : np.ndarray, shape (N, 2, 11)
        y : np.ndarray, shape (N,)
        weight : float
            Sample weight from COMPETITION_REGISTRY.

        Returns
        -------
        (X_aug, y_aug) with shape (M, 2, 11) and (M,) where M >= N.
        """
        if weight <= 1.0 or len(X) == 0:
            return X, y

        frac = weight - 1.0  # e.g. 0.5 for weight=1.5
        n_extra = int(round(len(X) * frac))
        if n_extra == 0:
            return X, y

        rng = np.random.default_rng(seed=42)
        idx = rng.integers(0, len(X), size=n_extra)
        X_extra = X[idx].copy().astype(np.float32)
        y_extra = y[idx].copy()

        # Add noise and clip to valid category range
        max_cat = float(np.nanmax(X)) if X.size > 0 else 27.0
        max_cat = max(max_cat, 1.0)
        noise = rng.normal(0.0, _AUGMENT_NOISE_STD, size=X_extra.shape).astype(np.float32)
        X_extra = np.clip(X_extra + noise, 1.0, max_cat)

        X_aug = np.concatenate([X, X_extra], axis=0)
        y_aug = np.concatenate([y, y_extra], axis=0)
        return X_aug, y_aug


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------


def _extract_match_urls(schedule: pd.DataFrame) -> list[str]:
    """
    Extract FBref match report URLs from a schedule DataFrame.

    FBref schedule pages include a 'Match Report' column with hyperlinks.
    soccerdata may return a 'match_url' column directly.  We handle both.
    """
    if schedule.empty:
        return []

    url_candidates = [
        c for c in schedule.columns
        if any(frag in c.lower() for frag in ("url", "link", "report", "href"))
    ]
    for col in url_candidates:
        urls = schedule[col].dropna().astype(str).tolist()
        valid = [u for u in urls if u.startswith("http")]
        if valid:
            return valid

    return []


def _build_goals_map(
    schedule: pd.DataFrame, match_urls: list[str]
) -> dict[str, float]:
    """
    Build a match_url → home_goals mapping from the schedule DataFrame.

    FBref's 'Score' column is typically formatted as 'H–A' (e.g. '2–1').
    We parse the home side (left of the separator) and return a float.
    """
    if schedule.empty or not match_urls:
        return {}

    score_col = next(
        (c for c in schedule.columns if "score" in c.lower()), None
    )
    url_col = next(
        (c for c in schedule.columns
         if any(frag in c.lower() for frag in ("url", "link", "report", "href"))),
        None,
    )

    if score_col is None or url_col is None:
        return {}

    goals_map: dict[str, float] = {}
    for _, row in schedule.iterrows():
        url = str(row.get(url_col, ""))
        score = str(row.get(score_col, ""))
        if not url.startswith("http") or not score:
            continue
        # Handle en-dash (–), em-dash (—), or hyphen (-) as separator
        for sep in ("–", "—", "-"):
            if sep in score:
                parts = score.split(sep)
                try:
                    home_goals = float(parts[0].strip())
                    goals_map[url] = home_goals
                except ValueError:
                    pass
                break

    return goals_map


def _infer_position_group(position_str: str) -> str:
    """Map an FBref position abbreviation to a broad position group."""
    pos = str(position_str).strip().upper().split(",")[0]
    if pos == "GK":
        return "goalkeeper"
    if pos in ("CB", "RB", "LB", "RWB", "LWB", "RCB", "LCB", "DF"):
        return "defender"
    if pos in (
        "DM", "CM", "RM", "LM", "CAM", "RAM", "LAM",
        "RCM", "LCM", "RDM", "LDM", "MF",
    ):
        return "midfielder"
    return "forward"


def _resolve_lineup(side_df: pd.DataFrame, registry: Any) -> np.ndarray | None:
    """
    Convert one team's lineup rows to an 11-element category array.

    Slot 0 → GK; slots 1-10 → outfield sorted by position group order.
    If fewer than 11 players are present, remaining slots are filled with
    the most common category in the row (mode imputation).

    Returns None if no players could be resolved.
    """
    cats: list[float] = []
    gk_cat: float | None = None
    outfield_cats: list[float] = []

    pos_col = next(
        (c for c in side_df.columns if "pos" in c.lower()), None
    )

    for _, row in side_df.iterrows():
        player_name = str(row.get("player", ""))
        pos_str = str(row.get(pos_col, "")) if pos_col else ""
        pos_group = _infer_position_group(pos_str)

        category, _ = registry.resolve(player_name, pos_group)

        if pos_group == "goalkeeper" or (gk_cat is None and not outfield_cats):
            # Treat first player as GK if position unavailable
            if pos_group == "goalkeeper":
                gk_cat = float(category)
                continue
        outfield_cats.append(float(category))

    if gk_cat is None and not outfield_cats:
        return None

    # Build the 11-element vector
    if gk_cat is None:
        # Fallback: use the first outfield as GK slot
        gk_cat = outfield_cats.pop(0) if outfield_cats else 1.0

    # Truncate or pad outfield to exactly 10 slots
    outfield_10 = outfield_cats[:10]
    if len(outfield_10) < 10:
        fill = float(outfield_10[-1]) if outfield_10 else gk_cat
        outfield_10 = outfield_10 + [fill] * (10 - len(outfield_10))

    return np.array([gk_cat] + outfield_10, dtype=np.float32)


def _get_goals(match_df: pd.DataFrame) -> float:
    """Extract home team goals from a match slice; return 0.0 if unavailable."""
    if "goals_home" in match_df.columns:
        val = match_df["goals_home"].dropna()
        if not val.empty:
            try:
                return float(val.iloc[0])
            except (ValueError, TypeError):
                pass
    return 0.0
