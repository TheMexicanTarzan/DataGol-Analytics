"""
Betting odds scrapers for upcoming and historical matches.

Two sources:

  OddsAPIScraper      → live / upcoming odds via The Odds API (https://the-odds-api.com)
                        Free tier: 500 requests/month — use the cache aggressively.
                        Requires ODDS_API_KEY environment variable.

  FootballDataScraper → historical odds from football-data.co.uk (free CSV downloads,
                        no authentication required).  Primary input for backtesting.

Odds are always returned in decimal (European) format.
"""

import logging
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .cache import DataCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared headers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# The Odds API sport keys per competition
ODDS_API_SPORT_KEYS: dict[str, str] = {
    "world_cup_2026":         "soccer_fifa_world_cup",
    "copa_america_2024":      "soccer_conmebol_copa_america",
    "euro_2024":              "soccer_uefa_european_championship",
    "champions_league_2324":  "soccer_uefa_champs_league",
    "premier_league_2324":    "soccer_epl",
    "bundesliga_2324":        "soccer_germany_bundesliga",
    "la_liga_2324":           "soccer_spain_la_liga",
}

# football-data.co.uk CSV URLs per competition
FOOTBALL_DATA_URLS: dict[str, str] = {
    "premier_league_2324":    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "bundesliga_2324":        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
    "la_liga_2324":           "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "champions_league_2324":  "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "world_cup_2022":         "https://www.football-data.co.uk/mmz4281/2223/WC2022.csv",
    "euro_2024":              "https://www.football-data.co.uk/mmz4281/2324/EURO2024.csv",
}

# Empty-DataFrame column schemas (returned on failure / missing data)
_ODDS_API_COLUMNS = [
    "match_id", "commence_time", "home_team", "away_team",
    "home_odds", "draw_odds", "away_odds",
    "home_implied_prob", "draw_implied_prob", "away_implied_prob",
    "bookmaker_margin",
]

_FOOTBALL_DATA_COLUMNS = [
    "date", "home_team", "away_team",
    "home_odds", "draw_odds", "away_odds",
    "home_goals", "away_goals", "outcome",
]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def remove_bookmaker_margin(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """
    Convert raw decimal odds to fair (margin-free) implied probabilities.

    raw_prob = 1 / odds
    fair_prob = raw_prob / sum(raw_probs)   ← removes the overround

    Returns (fair_home_prob, fair_draw_prob, fair_away_prob) summing to 1.0.
    """
    rp = [1 / home_odds, 1 / draw_odds, 1 / away_odds]
    total = sum(rp)
    return tuple(p / total for p in rp)


# ---------------------------------------------------------------------------
# OddsAPIScraper
# ---------------------------------------------------------------------------


class OddsAPIScraper:
    """
    Fetches upcoming match odds from The Odds API.

    API key must be set in the ODDS_API_KEY environment variable.
    Free tier: 500 requests/month — use the cache aggressively.

    Odds are returned in decimal format (European). The bookmaker consensus
    (average across all available bookmakers) is used to reduce house margin.
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 1,
        request_delay: float = 1.0,
    ):
        self._cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        self._delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upcoming_odds(self, competition: str) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
          match_id, commence_time, home_team, away_team,
          home_odds, draw_odds, away_odds,
          home_implied_prob, draw_implied_prob, away_implied_prob,
          bookmaker_margin

        Odds are consensus averages across all bookmakers in the response.
        bookmaker_margin = sum(implied_probs) - 1  (the vig; typically 0.04-0.08)
        """
        cache_key = f"odds_api_{competition}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        sport_key = ODDS_API_SPORT_KEYS.get(competition)
        if sport_key is None:
            logger.warning(
                "No Odds API sport key for '%s'. Available: %s",
                competition,
                list(ODDS_API_SPORT_KEYS),
            )
            return pd.DataFrame(columns=_ODDS_API_COLUMNS)

        try:
            api_key = self._get_api_key()
        except ValueError as exc:
            logger.warning("%s", exc)
            return pd.DataFrame(columns=_ODDS_API_COLUMNS)

        url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        logger.info("Fetching Odds API upcoming odds for %s", competition)
        time.sleep(self._delay)
        try:
            resp = self._session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            logger.warning("Odds API request failed for %s: %s", competition, exc)
            return pd.DataFrame(columns=_ODDS_API_COLUMNS)

        records = []
        for event in events:
            parsed = self._parse_bookmaker_odds(event)
            if parsed is None:
                continue

            h_prob = 1 / parsed["home_odds"]
            d_prob = 1 / parsed["draw_odds"]
            a_prob = 1 / parsed["away_odds"]
            margin = h_prob + d_prob + a_prob - 1.0

            records.append(
                {
                    "match_id":           event.get("id", ""),
                    "commence_time":      event.get("commence_time", ""),
                    "home_team":          event.get("home_team", ""),
                    "away_team":          event.get("away_team", ""),
                    "home_odds":          parsed["home_odds"],
                    "draw_odds":          parsed["draw_odds"],
                    "away_odds":          parsed["away_odds"],
                    "home_implied_prob":  round(h_prob, 4),
                    "draw_implied_prob":  round(d_prob, 4),
                    "away_implied_prob":  round(a_prob, 4),
                    "bookmaker_margin":   round(margin, 4),
                }
            )

        df = pd.DataFrame(records) if records else pd.DataFrame(columns=_ODDS_API_COLUMNS)
        if not df.empty:
            df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
            self._cache.set(cache_key, df)

        logger.info(
            "Odds API: %d upcoming matches fetched for %s", len(df), competition
        )
        return df

    def get_match_odds(
        self, home_team: str, away_team: str, competition: str
    ) -> dict | None:
        """
        Find odds for a specific match by team names (case-insensitive fuzzy match).

        Returns a dict with home_odds, draw_odds, away_odds, fair_home_prob,
        fair_draw_prob, fair_away_prob (margin-removed implied probabilities).
        Returns None if the match is not found.
        """
        df = self.get_upcoming_odds(competition)
        if df.empty:
            return None

        home_lower = home_team.lower()
        away_lower = away_team.lower()

        # Exact match first
        mask = (
            df["home_team"].str.lower() == home_lower
        ) & (
            df["away_team"].str.lower() == away_lower
        )
        if not mask.any():
            # Fuzzy: substring containment
            mask = (
                df["home_team"].str.lower().str.contains(home_lower, regex=False)
            ) & (
                df["away_team"].str.lower().str.contains(away_lower, regex=False)
            )

        if not mask.any():
            logger.warning(
                "No Odds API match found for %s vs %s in %s",
                home_team,
                away_team,
                competition,
            )
            return None

        row = df[mask].iloc[0]
        fair_h, fair_d, fair_a = remove_bookmaker_margin(
            float(row["home_odds"]),
            float(row["draw_odds"]),
            float(row["away_odds"]),
        )
        return {
            "home_odds":      float(row["home_odds"]),
            "draw_odds":      float(row["draw_odds"]),
            "away_odds":      float(row["away_odds"]),
            "fair_home_prob": round(fair_h, 4),
            "fair_draw_prob": round(fair_d, 4),
            "fair_away_prob": round(fair_a, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        key = os.environ.get("ODDS_API_KEY", "")
        if not key:
            raise ValueError(
                "ODDS_API_KEY environment variable not set. "
                "Get a free key at https://the-odds-api.com"
            )
        return key

    def _parse_bookmaker_odds(self, event: dict) -> dict | None:
        """
        Average h2h (3-way: home/draw/away) odds across all bookmakers.
        Returns {home_odds, draw_odds, away_odds} or None if no h2h market found.

        Outcome names are matched to home/away team names via case-insensitive
        substring matching so minor API name discrepancies don't cause failures.
        """
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        bookmakers = event.get("bookmakers", [])

        home_prices: list[float] = []
        draw_prices: list[float] = []
        away_prices: list[float] = []

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                bk_home = bk_draw = bk_away = None

                for outcome in outcomes:
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    if price is None:
                        continue
                    price = float(price)

                    name_lower = name.lower()
                    if name_lower == "draw":
                        bk_draw = price
                    elif (
                        name_lower == home_team.lower()
                        or home_team.lower() in name_lower
                        or name_lower in home_team.lower()
                    ):
                        bk_home = price
                    elif (
                        name_lower == away_team.lower()
                        or away_team.lower() in name_lower
                        or name_lower in away_team.lower()
                    ):
                        bk_away = price

                if bk_home is not None:
                    home_prices.append(bk_home)
                if bk_draw is not None:
                    draw_prices.append(bk_draw)
                if bk_away is not None:
                    away_prices.append(bk_away)
                break  # one h2h market per bookmaker

        if not home_prices or not draw_prices or not away_prices:
            return None

        return {
            "home_odds": round(sum(home_prices) / len(home_prices), 3),
            "draw_odds": round(sum(draw_prices) / len(draw_prices), 3),
            "away_odds": round(sum(away_prices) / len(away_prices), 3),
        }


# ---------------------------------------------------------------------------
# FootballDataScraper
# ---------------------------------------------------------------------------


class FootballDataScraper:
    """
    Downloads historical match odds from football-data.co.uk (free CSVs, no auth).

    Returns standardised DataFrames with columns:
      date, home_team, away_team,
      home_odds, draw_odds, away_odds,   <- Bet365 odds (B365H, B365D, B365A)
      home_goals, away_goals, outcome    <- 'H', 'D', or 'A'

    These DataFrames are the primary input for backtesting.
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 720,
        request_delay: float = 2.0,
    ):
        self._cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        self._delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(
            {
                **_HEADERS,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical_odds(self, competition: str) -> pd.DataFrame:
        """
        Download and parse odds CSV for the competition.

        Returns standardised DataFrame (see class docstring for columns).
        Rows with missing odds or scores are dropped.
        """
        cache_key = f"football_data_{competition}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = FOOTBALL_DATA_URLS.get(competition)
        if url is None:
            logger.warning(
                "No football-data.co.uk URL for '%s'. Available: %s",
                competition,
                list(FOOTBALL_DATA_URLS),
            )
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        logger.info("Fetching football-data.co.uk CSV for %s: %s", competition, url)
        time.sleep(self._delay)
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            raw_csv = resp.text
        except Exception as exc:
            logger.warning(
                "football-data.co.uk download failed for %s: %s", competition, exc
            )
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        df = self._parse_csv(raw_csv)
        if df.empty:
            logger.warning(
                "No usable rows in football-data.co.uk CSV for %s", competition
            )
            return df

        self._cache.set(cache_key, df)
        logger.info(
            "football-data.co.uk: %d matches loaded for %s", len(df), competition
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_csv(self, raw_csv: str) -> pd.DataFrame:
        """
        Parse the raw football-data.co.uk CSV.

        Key column mappings:
          Date   -> date       (datetime; handles DD/MM/YY and YYYY-MM-DD)
          HomeTeam -> home_team
          AwayTeam -> away_team
          B365H  -> home_odds   (Bet365 home win odds, decimal)
          B365D  -> draw_odds
          B365A  -> away_odds
          FTHG   -> home_goals  (Full-Time Home Goals)
          FTAG   -> away_goals
          FTR    -> outcome ('H'/'D'/'A')

        If B365* columns are absent, falls back to BbAvH/BbAvD/BbAvA
        (Betbrain average). If neither is present, returns empty DataFrame.
        """
        try:
            df = pd.read_csv(StringIO(raw_csv), low_memory=False)
        except Exception as exc:
            logger.warning("CSV parse error: %s", exc)
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        # Drop completely empty rows that football-data CSVs often trail with
        df = df.dropna(how="all").reset_index(drop=True)

        if df.empty:
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        # -- Odds columns: Bet365 preferred, Betbrain average as fallback ------
        if all(c in df.columns for c in ("B365H", "B365D", "B365A")):
            odds_home_col, odds_draw_col, odds_away_col = "B365H", "B365D", "B365A"
        elif all(c in df.columns for c in ("BbAvH", "BbAvD", "BbAvA")):
            odds_home_col, odds_draw_col, odds_away_col = "BbAvH", "BbAvD", "BbAvA"
            logger.info("B365 columns absent; using Betbrain average odds")
        else:
            logger.warning(
                "No recognised odds columns (B365* or BbAv*) in CSV. "
                "Available columns: %s",
                list(df.columns),
            )
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        # -- Required result columns -------------------------------------------
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(
                "football-data CSV missing required columns: %s", missing
            )
            return pd.DataFrame(columns=_FOOTBALL_DATA_COLUMNS)

        # -- Build output DataFrame --------------------------------------------
        out = pd.DataFrame()
        out["home_team"]  = df["HomeTeam"].astype(str).str.strip()
        out["away_team"]  = df["AwayTeam"].astype(str).str.strip()
        out["home_odds"]  = pd.to_numeric(df[odds_home_col], errors="coerce")
        out["draw_odds"]  = pd.to_numeric(df[odds_draw_col], errors="coerce")
        out["away_odds"]  = pd.to_numeric(df[odds_away_col], errors="coerce")
        out["home_goals"] = pd.to_numeric(df["FTHG"], errors="coerce")
        out["away_goals"] = pd.to_numeric(df["FTAG"], errors="coerce")
        out["outcome"]    = df["FTR"].astype(str).str.strip()

        # -- Date parsing: try DD/MM/YY first, then YYYY-MM-DD ----------------
        raw_dates = df["Date"].astype(str).str.strip()
        parsed_dates = pd.to_datetime(raw_dates, format="%d/%m/%y", errors="coerce")
        still_nat = parsed_dates.isna()
        if still_nat.any():
            parsed_dates[still_nat] = pd.to_datetime(
                raw_dates[still_nat], format="%d/%m/%Y", errors="coerce"
            )
        still_nat = parsed_dates.isna()
        if still_nat.any():
            parsed_dates[still_nat] = pd.to_datetime(
                raw_dates[still_nat], errors="coerce"
            )
        out["date"] = parsed_dates

        # -- Drop rows with missing odds or match result -----------------------
        out = out.dropna(
            subset=["date", "home_odds", "draw_odds", "away_odds", "home_goals", "away_goals"]
        )
        out = out[out["outcome"].isin(["H", "D", "A"])]
        out = out.reset_index(drop=True)

        return out[_FOOTBALL_DATA_COLUMNS]
