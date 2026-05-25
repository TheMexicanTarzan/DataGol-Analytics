"""
DataGol data pipeline — drop-in replacement for the StatsBomb API calls.

Produces the same two objects that the DataGol notebook uses downstream:
  - model_df          : player feature matrix (for clustering)
  - copa_america_lineups : lineup DataFrame (for matchup prediction)

Usage (notebook cell):
    from src.pipeline import load_tournament_data
    model_df, lineups_df = load_tournament_data("world_cup_2026")
"""

import logging
from pathlib import Path

import pandas as pd

from .data import DataCache, FBrefScraper, PersonalityFeatureBuilder

logger = logging.getLogger(__name__)


def load_tournament_data(
    competition: str = "world_cup_2026",
    cache_dir: Path = Path("data/cache"),
    ttl_hours: int = 6,
    min_minutes: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch, clean, and featurise all player data for the given tournament.

    Parameters
    ----------
    competition : str
        One of: world_cup_2026, world_cup_2022, copa_america_2024, euro_2024
    cache_dir : Path
        Where to store cached parquet files (avoids re-scraping).
    ttl_hours : int
        Cache time-to-live. Use 6 during the tournament, 720 for historical.
    min_minutes : int
        Minimum minutes played to include a player.

    Returns
    -------
    model_df : pd.DataFrame
        Feature matrix (players × personality features), ready for clustering.
    lineups_df : pd.DataFrame
        Long-format lineups: columns [match_id, team, player, position].
    """
    scraper = FBrefScraper(
        competition=competition,
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
    )
    builder = PersonalityFeatureBuilder()
    builder.MIN_MINUTES = min_minutes

    logger.info("Fetching player stats for '%s' …", competition)
    merged_stats = scraper.get_merged_player_stats()

    logger.info("Building personality features …")
    model_df = builder.build(merged_stats)

    logger.info("Fetching match schedule …")
    schedule = scraper.get_schedule()

    logger.info("Fetching match lineups …")
    match_urls = _extract_match_urls(schedule)
    lineups_df = scraper.get_all_lineups(match_urls)

    logger.info(
        "Pipeline complete: %d players, %d lineup records.",
        len(model_df),
        len(lineups_df),
    )
    return model_df, lineups_df


def _extract_match_urls(schedule: pd.DataFrame) -> list[str]:
    """Extract FBref match report URLs from the schedule DataFrame."""
    url_col = next(
        (c for c in schedule.columns if "url" in c.lower() or "report" in c.lower()),
        None,
    )
    if url_col is None:
        logger.warning("No match URL column found in schedule. Lineup scraping skipped.")
        return []

    base = "https://fbref.com"
    urls = schedule[url_col].dropna().tolist()
    return [u if u.startswith("http") else f"{base}{u}" for u in urls]
