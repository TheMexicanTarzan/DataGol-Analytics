"""
API-Football v3 scraper — fetches player stats, schedules, and lineups.

Requires API_FOOTBALL_KEY environment variable (loaded from .env via python-dotenv).
Authentication: x-apisports-key header.

Supported competition keys and their (league_id, season) tuples are in
API_FOOTBALL_COMPETITION_MAP.  Column ordering in get_merged_player_stats()
output is identical to StatsBombScraper so PersonalityFeatureBuilder._col()
works without modification.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .cache import DataCache

logger = logging.getLogger(__name__)

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

API_FOOTBALL_COMPETITION_MAP: dict[str, tuple[int, int]] = {
    "world_cup_2026":        (1,   2026),
    "world_cup_2022":        (1,   2022),
    "copa_america_2024":     (9,   2024),
    "euro_2024":             (4,   2024),
    "champions_league_2425": (2,   2024),
    "champions_league_2324": (2,   2023),
    "premier_league_2425":   (39,  2024),
    "premier_league_2324":   (39,  2023),
    "bundesliga_2425":       (78,  2024),
    "bundesliga_2324":       (78,  2023),
    "la_liga_2425":          (140, 2024),
    "la_liga_2324":          (140, 2023),
}

_AF_POS_MAP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender":   "DF",
    "Midfielder": "MF",
    "Attacker":   "FW",
}


class APIFootballScraper:
    """
    Fetches player stats, schedules, and lineups from API-Football v3.

    get_merged_player_stats() returns a DataFrame whose column names and order
    are compatible with PersonalityFeatureBuilder._col() — identical ordering
    to StatsBombScraper so both can be used interchangeably.

    Free-tier rate limit: 100 requests/day, ~30 requests/minute.
    request_delay defaults to 1.2s to stay well within the per-minute limit.
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 6,
        request_delay: float = 1.2,
    ):
        self._cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        self._delay = request_delay
        self._session = requests.Session()
        key = os.environ.get("API_FOOTBALL_KEY", "")
        if key:
            self._session.headers.update({
                "x-apisports-key": key,
                "Accept": "application/json",
            })
        self._has_key = bool(key)

    def is_supported(self, competition: str) -> bool:
        return competition in API_FOOTBALL_COMPETITION_MAP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_merged_player_stats(self, competition: str) -> pd.DataFrame:
        """
        Fetch all player statistics for the competition.
        Paginates through /players (20 players/page) and aggregates.

        Returns a PersonalityFeatureBuilder-compatible DataFrame.
        """
        self._require_key()
        cache_key = f"{competition}_api_football_player_stats"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("API-Football: loaded %s from cache (%d players).", competition, len(cached))
            return cached

        if competition not in API_FOOTBALL_COMPETITION_MAP:
            raise ValueError(f"API-Football: competition '{competition}' not in map.")

        league_id, season = API_FOOTBALL_COMPETITION_MAP[competition]
        rows = self._fetch_all_players(league_id, season)
        if not rows:
            raise ValueError(f"API-Football: no player data returned for {competition}.")

        df = pd.DataFrame(rows)
        self._cache.set(cache_key, df)
        logger.info("API-Football: cached stats for %d players in %s.", len(df), competition)
        return df

    def get_schedule(self, competition: str) -> pd.DataFrame:
        """
        Fetch the full fixture list for the competition.

        Returns DataFrame columns: fixture_id, date, home_team, away_team, status.
        """
        self._require_key()
        cache_key = f"{competition}_api_football_schedule"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        league_id, season = API_FOOTBALL_COMPETITION_MAP[competition]
        data = self._get(f"{API_FOOTBALL_BASE}/fixtures", {"league": league_id, "season": season})

        rows = []
        for fix in data.get("response", []):
            f = fix.get("fixture", {})
            teams = fix.get("teams", {})
            rows.append({
                "fixture_id": f.get("id"),
                "date":       (f.get("date") or "")[:10],
                "home_team":  (teams.get("home") or {}).get("name", ""),
                "away_team":  (teams.get("away") or {}).get("name", ""),
                "status":     (f.get("status") or {}).get("short", ""),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            self._cache.set(cache_key, df)
        return df

    def get_match_lineups(self, fixture_id: int) -> pd.DataFrame:
        """
        Fetch the starting XI for both sides of a fixture.

        Returns long-format DataFrame: fixture_id, side, team, player, position, grid.
        """
        self._require_key()
        data = self._get(f"{API_FOOTBALL_BASE}/fixtures/lineups", {"fixture": fixture_id})
        rows = []
        for i, team_lineup in enumerate(data.get("response", [])):
            side = "home" if i == 0 else "away"
            team_name = (team_lineup.get("team") or {}).get("name", "")
            for player_entry in team_lineup.get("startXI", []):
                p = (player_entry.get("player") or {})
                rows.append({
                    "fixture_id": fixture_id,
                    "side":       side,
                    "team":       team_name,
                    "player":     p.get("name", ""),
                    "position":   p.get("pos", ""),
                    "grid":       p.get("grid", ""),
                })
        return pd.DataFrame(rows)

    def get_all_lineups(self, fixture_ids: list[int]) -> pd.DataFrame:
        """Batch-fetch lineups for a list of fixture IDs."""
        frames = []
        for fid in fixture_ids:
            try:
                df = self.get_match_lineups(fid)
                if not df.empty:
                    frames.append(df)
                time.sleep(self._delay)
            except Exception as exc:
                logger.warning("API-Football: lineup fetch failed for fixture %d — %s", fid, exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_key(self) -> None:
        if not self._has_key:
            raise EnvironmentError(
                "API_FOOTBALL_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )

    def _fetch_all_players(self, league_id: int, season: int) -> list[dict]:
        url = f"{API_FOOTBALL_BASE}/players"
        page, rows = 1, []
        while True:
            data = self._get(url, {"league": league_id, "season": season, "page": page})
            response = data.get("response", [])
            if not response:
                break
            for entry in response:
                row = self._parse_player_entry(entry)
                if row is not None:
                    rows.append(row)
            paging = data.get("paging", {})
            total_pages = paging.get("total", 1)
            logger.debug("API-Football: page %d/%d (%d players so far)", page, total_pages, len(rows))
            if page >= total_pages:
                break
            page += 1
            time.sleep(self._delay)
        return rows

    def _parse_player_entry(self, entry: dict) -> dict | None:
        """
        Map a single API-Football /players response entry to the column ordering
        required by PersonalityFeatureBuilder._col().

        Column order mirrors StatsBombScraper._aggregate_events() — shorter
        fragment columns come before any column whose name contains them as a
        substring, so _col()'s first-match logic returns the right column.
        """
        player = entry.get("player") or {}
        stats_list = entry.get("statistics") or []
        if not stats_list:
            return None

        name = player.get("name", "")
        if not name:
            return None

        raw_pos = _get_position(stats_list)
        pos     = _AF_POS_MAP.get(raw_pos, "MF")
        minutes = _agg(stats_list, ["games", "minutes"])

        pass_att   = _agg(stats_list, ["passes", "total"])
        accuracy   = _pct(stats_list, ["passes", "accuracy"]) / 100.0
        pass_cmp   = pass_att * accuracy

        shots      = _agg(stats_list, ["shots", "total"])
        goals      = _agg(stats_list, ["goals", "total"])
        assists    = _agg(stats_list, ["goals", "assists"])
        key_passes = _agg(stats_list, ["passes", "key"])
        tkl_total  = _agg(stats_list, ["tackles", "total"])
        tkl_won    = _agg(stats_list, ["duels", "won"])
        blocks     = _agg(stats_list, ["tackles", "blocks"])
        inter      = _agg(stats_list, ["tackles", "interceptions"])
        drib_att   = _agg(stats_list, ["dribbles", "attempts"])
        drib_succ  = _agg(stats_list, ["dribbles", "success"])
        drib_past  = _agg(stats_list, ["dribbles", "past"])
        fouls      = _agg(stats_list, ["fouls", "committed"])
        yellow     = _agg(stats_list, ["cards", "yellow"])
        red        = _agg(stats_list, ["cards", "red"]) + _agg(stats_list, ["cards", "yellowred"])

        return {
            "Player": name,
            "Pos":    pos,
            "Min":    minutes,
            # Fragment-ordering required by _col() — shorter fragments first
            "tkl":          tkl_total,
            "tklw":         tkl_won,
            "sh":           shots,
            "gls":          goals,
            "xg":           0.0,        # API-Football v3 does not expose xG
            "ast":          assists,
            "int":          inter,
            "kp":           key_passes,
            "crs":          0.0,
            "clr":          0.0,
            "fls":          fouls,
            "crdy":         yellow,
            "crdr":         red,
            "ck":           0.0,
            "dribbled_past": drib_past,
            "aerialwon":    0.0,
            "aeriallost":   0.0,
            "block_blocks": blocks,
            "block_sh":     0.0,
            "block_pass":   0.0,
            "pass_att":     pass_att,
            "pass_cmp":     pass_cmp,
            "total_cmp":    pass_cmp,
            "short_att":    0.0,
            "short_cmp":    0.0,
            "medium_att":   0.0,
            "long_att":     0.0,
            "corner":       0.0,
            "ti":           0.0,
            "dead":         0.0,
            "npxg_sh":      0.0,
            "shot_aerial":  0.0,
            "dribble_att":  drib_att,
            "dribble_succ": drib_succ,
        }

    def _get(self, url: str, params: dict) -> dict:
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("API-Football: request failed %s — %s", url, exc)
            return {}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_position(stats_list: list[dict]) -> str:
    for s in stats_list:
        pos = (s.get("games") or {}).get("position")
        if pos:
            return str(pos)
    return "Midfielder"


def _agg(stats_list: list[dict], path: list[str]) -> float:
    """Sum a nested field across all statistics entries (handles multiple clubs)."""
    total = 0.0
    for s in stats_list:
        node = s
        for key in path:
            node = (node or {}).get(key)
        try:
            total += float(node or 0)
        except (TypeError, ValueError):
            pass
    return total


def _pct(stats_list: list[dict], path: list[str]) -> float:
    """Read a percentage field — API-Football returns it as a numeric string e.g. '89'."""
    for s in stats_list:
        node = s
        for key in path:
            node = (node or {}).get(key)
        if node is not None:
            try:
                return float(str(node).replace("%", ""))
            except ValueError:
                pass
    return 0.0
