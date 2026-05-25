"""
FBref scraper for tournament player stats and match lineups.

Strategy:
  Primary  → soccerdata library (handles FBref structure changes, caching)
  Fallback → Direct HTML parsing via requests + pandas.read_html

FBref comments out some stat tables in the HTML for non-subscribers.
The _uncomment_html helper strips those comment wrappers before parsing.
"""

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from .cache import DataCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Competition registry
# ---------------------------------------------------------------------------

CompetitionKey = Literal[
    "world_cup_2026",
    "world_cup_2022",
    "copa_america_2024",
    "euro_2024",
    "bundesliga_2024",
    "premier_league_2024",
]

# Direct FBref competition stats URLs (no auth required)
FBREF_COMP_URLS: dict[str, str] = {
    "world_cup_2026": "https://fbref.com/en/comps/1/stats/World-Cup-Stats",
    "world_cup_2022": "https://fbref.com/en/comps/1/2022/2022-World-Cup-Stats",
    "copa_america_2024": "https://fbref.com/en/comps/685/2024/2024-Copa-America-Stats",
    "euro_2024": "https://fbref.com/en/comps/676/2024/2024-UEFA-Euro-Stats",
}

# soccerdata (FBref wrapper) identifiers
SOCCERDATA_MAP: dict[str, tuple[str, str]] = {
    "world_cup_2026": ("FIFA World Cup", "2026"),
    "world_cup_2022": ("FIFA World Cup", "2022"),
    "copa_america_2024": ("Copa América", "2024"),
    "euro_2024": ("UEFA Euro", "2024"),
    "bundesliga_2024": ("Bundesliga", "2024"),
    "premier_league_2024": ("Premier League", "2024"),
}

# Table IDs on FBref stats pages (used in the direct HTML scraper)
STAT_TABLE_IDS: dict[str, str] = {
    "standard": "stats_standard",
    "shooting": "stats_shooting",
    "passing": "stats_passing",
    "passing_types": "stats_passing_types",
    "defense": "stats_defense",
    "possession": "stats_possession",
    "misc": "stats_misc",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _uncomment_html(html: str) -> str:
    """FBref hides some tables inside HTML comments. Strip those wrappers."""
    soup = BeautifulSoup(html, "lxml")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            comment.replace_with(BeautifulSoup(comment, "lxml"))
    return str(soup)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-level column headers from FBref tables into flat strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(c) for c in col if c and not str(c).startswith("Unnamed")
            ).strip("_")
            for col in df.columns.values
        ]
    return df


def _clean_player_table(df: pd.DataFrame) -> pd.DataFrame:
    """Remove summary rows (like 'Squad Total') and reset index."""
    df = _flatten_columns(df)
    df.columns = df.columns.str.strip()

    # FBref repeats the header row every N rows and adds Squad/Nation totals
    if "Player" in df.columns:
        df = df[df["Player"].notna() & (df["Player"] != "Player")]
        df = df[~df["Player"].str.contains(r"^Squad|^vs\.", na=False, regex=True)]

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Direct HTML scraper (primary fallback)
# ---------------------------------------------------------------------------


class _DirectFBrefScraper:
    """Scrapes FBref competition pages directly without third-party libraries."""

    def __init__(self, request_delay: float = 3.0):
        self._delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get(self, url: str) -> str:
        time.sleep(self._delay)
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _scrape_stat_table(self, base_url: str, stat: str) -> pd.DataFrame:
        table_id = STAT_TABLE_IDS[stat]
        html = self._get(base_url)
        html = _uncomment_html(html)
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": table_id})
        if table is None:
            raise ValueError(f"Table '{table_id}' not found at {base_url}")
        df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]
        return _clean_player_table(df)

    def get_player_stats(self, competition: str) -> dict[str, pd.DataFrame]:
        """Return a dict of stat_type → DataFrame for all FBref stat tables."""
        base_url = FBREF_COMP_URLS.get(competition)
        if base_url is None:
            raise ValueError(
                f"Unknown competition '{competition}'. "
                f"Available: {list(FBREF_COMP_URLS)}"
            )
        stats: dict[str, pd.DataFrame] = {}
        for stat in STAT_TABLE_IDS:
            logger.info("Scraping FBref '%s' stats for %s …", stat, competition)
            try:
                stats[stat] = self._scrape_stat_table(base_url, stat)
            except Exception as exc:
                logger.warning("Skipping '%s' stats: %s", stat, exc)
        return stats

    def get_schedule(self, competition: str) -> pd.DataFrame:
        """Scrape match schedule/results table."""
        base_url = FBREF_COMP_URLS.get(competition, "")
        schedule_url = base_url.replace("stats/", "schedule/").replace("-Stats", "-Schedule")
        html = self._get(schedule_url)
        html = _uncomment_html(html)
        dfs = pd.read_html(StringIO(html))
        # FBref schedule table is usually the first large one with 'Score' column
        for df in dfs:
            flat = _flatten_columns(df)
            if "Score" in flat.columns or "score" in flat.columns.str.lower().tolist():
                return _clean_player_table(flat)
        raise ValueError(f"Schedule table not found for {competition}")

    def get_match_lineups(self, match_url: str) -> dict[str, pd.DataFrame]:
        """
        Scrape starting lineups from a single FBref match report page.
        Returns {'home': DataFrame, 'away': DataFrame} with columns
        [player, position, shirt_number].
        """
        html = self._get(match_url)
        html = _uncomment_html(html)
        soup = BeautifulSoup(html, "lxml")

        lineups: dict[str, pd.DataFrame] = {}
        lineup_divs = soup.find_all("div", class_="lineup")
        sides = ["home", "away"]

        for i, div in enumerate(lineup_divs[:2]):
            rows = []
            for tr in div.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                if len(cells) >= 2:
                    rows.append(cells)
            if rows:
                df = pd.DataFrame(rows, columns=["shirt_number", "player"][: len(rows[0])])
                # The first row is usually the GK, rest are outfield by position block
                lineups[sides[i]] = df

        if not lineups:
            raise ValueError(f"Lineup divs not found at {match_url}")
        return lineups


# ---------------------------------------------------------------------------
# soccerdata wrapper (preferred when installed)
# ---------------------------------------------------------------------------


class _SoccerdataFBrefScraper:
    """Thin wrapper around soccerdata.FBref for structured stat retrieval."""

    def __init__(self):
        try:
            import soccerdata as sd  # noqa: F401
            self._sd = sd
        except ImportError as exc:
            raise ImportError(
                "soccerdata is not installed. Run: pip install soccerdata"
            ) from exc

    def get_player_stats(self, competition: str) -> dict[str, pd.DataFrame]:
        league, season = SOCCERDATA_MAP[competition]
        fbref = self._sd.FBref(leagues=league, seasons=season)
        stat_types = [
            "standard", "shooting", "passing",
            "passing_types", "defense", "possession", "misc",
        ]
        stats: dict[str, pd.DataFrame] = {}
        for stat in stat_types:
            logger.info("soccerdata fetching '%s' for %s …", stat, competition)
            try:
                df = fbref.read_player_season_stats(stat_type=stat)
                stats[stat] = _clean_player_table(_flatten_columns(df.reset_index()))
            except Exception as exc:
                logger.warning("soccerdata failed for '%s': %s", stat, exc)
        return stats

    def get_schedule(self, competition: str) -> pd.DataFrame:
        league, season = SOCCERDATA_MAP[competition]
        fbref = self._sd.FBref(leagues=league, seasons=season)
        df = fbref.read_schedule()
        return _clean_player_table(_flatten_columns(df.reset_index()))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class FBrefScraper:
    """
    Tournament data scraper with automatic fallback chain:
      soccerdata → direct HTML (FBref)

    Usage
    -----
    scraper = FBrefScraper(competition="world_cup_2026")
    stats   = scraper.get_player_stats()   # dict of DataFrames per stat type
    schedule = scraper.get_schedule()      # match results
    lineup  = scraper.get_match_lineups(match_url)  # one match
    """

    def __init__(
        self,
        competition: CompetitionKey = "world_cup_2026",
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 6,
        request_delay: float = 3.0,
    ):
        self.competition = competition
        self._cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        self._direct = _DirectFBrefScraper(request_delay=request_delay)

        try:
            self._soccerdata = _SoccerdataFBrefScraper()
            logger.info("Using soccerdata as primary scraper.")
        except ImportError:
            self._soccerdata = None
            logger.info("soccerdata unavailable; using direct HTML scraper.")

    # ------------------------------------------------------------------

    def get_player_stats(self) -> dict[str, pd.DataFrame]:
        """
        Fetch all available FBref stat tables for the competition.
        Results are cached to disk; re-fetched when TTL expires.
        """
        cache_key = f"{self.competition}_player_stats"
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Cache stores the merged df; re-split is not needed — return as-is
            return {"merged": cached}

        if self._soccerdata and self.competition in SOCCERDATA_MAP:
            try:
                stats = self._soccerdata.get_player_stats(self.competition)
                merged = self._merge_stat_tables(stats)
                self._cache.set(cache_key, merged)
                return stats
            except Exception as exc:
                logger.warning("soccerdata failed, falling back to direct scraper: %s", exc)

        stats = self._direct.get_player_stats(self.competition)
        merged = self._merge_stat_tables(stats)
        self._cache.set(cache_key, merged)
        return stats

    def get_merged_player_stats(self) -> pd.DataFrame:
        """
        Convenience method that returns a single merged DataFrame.
        Player name is the join key across all stat tables.
        Cached independently for quick re-access.
        """
        cache_key = f"{self.competition}_player_stats"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = self.get_player_stats()
        merged = self._merge_stat_tables(raw)
        self._cache.set(cache_key, merged)
        return merged

    def get_schedule(self) -> pd.DataFrame:
        """Fetch match schedule and results for the competition."""
        cache_key = f"{self.competition}_schedule"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._soccerdata and self.competition in SOCCERDATA_MAP:
            try:
                df = self._soccerdata.get_schedule(self.competition)
                self._cache.set(cache_key, df)
                return df
            except Exception as exc:
                logger.warning("soccerdata schedule failed, trying direct: %s", exc)

        df = self._direct.get_schedule(self.competition)
        self._cache.set(cache_key, df)
        return df

    def get_match_lineups(self, match_url: str) -> dict[str, pd.DataFrame]:
        """
        Scrape starting lineups for a single match.
        match_url: full FBref match report URL
                   e.g. 'https://fbref.com/en/matches/abc123/...'
        """
        cache_key = f"lineup_{match_url.split('/')[-2]}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {"home": cached[cached["side"] == "home"].drop(columns="side"),
                    "away": cached[cached["side"] == "away"].drop(columns="side")}

        lineups = self._direct.get_match_lineups(match_url)

        # Persist as a single table
        frames = []
        for side, df in lineups.items():
            df = df.copy()
            df["side"] = side
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        self._cache.set(cache_key, combined)
        return lineups

    def get_all_lineups(self, match_report_urls: list[str]) -> pd.DataFrame:
        """
        Batch-fetch lineups for all provided match report URLs.
        Returns a long-format DataFrame with columns:
          match_url, side, player, position, shirt_number
        """
        records = []
        for url in match_report_urls:
            try:
                lineups = self.get_match_lineups(url)
                for side, df in lineups.items():
                    df = df.copy()
                    df["match_url"] = url
                    df["side"] = side
                    records.append(df)
                logger.info("Lineups fetched: %s", url)
            except Exception as exc:
                logger.warning("Failed to fetch lineups for %s: %s", url, exc)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_stat_tables(stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Left-join all stat tables on player name.
        Duplicate columns from different tables get a suffix to avoid collisions.
        """
        player_col = _find_player_col(stats)
        merged: pd.DataFrame | None = None
        for stat_type, df in stats.items():
            if df.empty:
                continue
            col = _find_player_col({stat_type: df})
            if col is None:
                continue
            df = df.copy()
            df = df.rename(columns={col: "player"})
            # Prefix stat-type onto all non-player columns to avoid collisions
            df = df.rename(
                columns={c: f"{stat_type}_{c}" for c in df.columns if c != "player"}
            )
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="player", how="outer")

        return merged if merged is not None else pd.DataFrame()


def _find_player_col(stats: dict[str, pd.DataFrame]) -> str | None:
    for df in stats.values():
        for candidate in ("Player", "player", "Name", "name"):
            if candidate in df.columns:
                return candidate
    return None
