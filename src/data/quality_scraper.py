"""
Player quality scoring from three complementary sources:

  Transfermarkt  → market value  (pre-tournament baseline, positional depth)
  FBref          → percentile ranks computed from existing stats (no extra requests)
  Sofascore      → per-match ratings (live signal; weight grows with appearances)

Combined score formula per player:

    quality = w_tm   * norm_tm_value
            + w_fbr  * norm_fbref_percentile
            + w_sof  * norm_sofascore_rating      (0 when appearances < min_apps)

All three components are normalised to [0, 1] within each position group
(defenders / midfielders / forwards) so a world-class CB isn't penalised for
having a lower market value than a world-class striker.

Weights are re-distributed automatically when Sofascore data is unavailable
for a player (e.g. before the tournament starts or after a group-stage exit).
"""

import logging
import time
import unicodedata
from difflib import get_close_matches
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .cache import DataCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_API_HEADERS = {
    **_HEADERS,
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json",
}

POSITION_GROUPS: dict[str, list[str]] = {
    "defender": ["CB", "RB", "LB", "RWB", "LWB", "D", "DF"],
    "midfielder": ["DM", "CM", "CAM", "RM", "LM", "AM", "M", "MF"],
    "forward": ["RW", "LW", "CF", "ST", "SS", "FW", "F", "A", "W"],
}

_POS_LOOKUP: dict[str, str] = {
    abbr: group
    for group, abbrs in POSITION_GROUPS.items()
    for abbr in abbrs
}


def _normalise_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode()
    return " ".join(ascii_name.lower().split())


def _match_player(name: str, target_names: list[str], cutoff: float = 0.8) -> str | None:
    """Return the closest match from target_names or None if below cutoff."""
    norm = _normalise_name(name)
    norm_targets = [_normalise_name(t) for t in target_names]
    matches = get_close_matches(norm, norm_targets, n=1, cutoff=cutoff)
    if not matches:
        return None
    idx = norm_targets.index(matches[0])
    return target_names[idx]


def _position_group(pos_str: str) -> str:
    """Map a raw position string to defender / midfielder / forward."""
    token = str(pos_str).strip().upper().split(",")[0].split("/")[0]
    return _POS_LOOKUP.get(token, "midfielder")


def _minmax_by_group(df: pd.DataFrame, value_col: str, pos_col: str) -> pd.Series:
    """Min-max normalise value_col within each position group. Returns [0, 1] Series."""
    result = pd.Series(np.nan, index=df.index)
    for group in ("defender", "midfielder", "forward"):
        mask = df[pos_col] == group
        if mask.sum() == 0:
            continue
        vals = df.loc[mask, value_col].astype(float)
        lo, hi = vals.min(), vals.max()
        if hi > lo:
            result.loc[mask] = (vals - lo) / (hi - lo)
        else:
            result.loc[mask] = 0.5  # all equal within group
    return result.fillna(0.5)


# ---------------------------------------------------------------------------
# Transfermarkt scraper
# ---------------------------------------------------------------------------

TRANSFERMARKT_COMP_URLS: dict[str, str] = {
    "world_cup_2026": (
        "https://www.transfermarkt.com/weltmeisterschaft-2026"
        "/marktwerteinsatz/pokalwettbewerb/WM26"
    ),
    "world_cup_2022": (
        "https://www.transfermarkt.com/weltmeisterschaft-2022"
        "/marktwerteinsatz/pokalwettbewerb/WM22"
    ),
    "copa_america_2024": (
        "https://www.transfermarkt.com/copa-america-2024"
        "/marktwerteinsatz/pokalwettbewerb/CAM4"
    ),
    "euro_2024": (
        "https://www.transfermarkt.com/europameisterschaft-2024"
        "/marktwerteinsatz/pokalwettbewerb/EM24"
    ),
}


def _parse_tm_value(raw: str) -> float:
    """'€50.00m' → 50_000_000.0  |  '€500k' → 500_000.0  |  '-' → 0.0"""
    s = str(raw).replace("€", "").replace(",", ".").strip()
    if not s or s in ("-", "n/a", "nan"):
        return 0.0
    if s.endswith("m"):
        return float(s[:-1]) * 1_000_000
    if s.endswith("k"):
        return float(s[:-1]) * 1_000
    try:
        return float(s)
    except ValueError:
        return 0.0


class TransfermarktScraper:
    """
    Scrapes the market value of every player in a competition from Transfermarkt.

    Transfermarkt is reasonably scraper-friendly for public pages, but requires
    a realistic User-Agent and a polite request delay.
    """

    def __init__(self, request_delay: float = 4.0):
        self._delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(
            {**_HEADERS, "Referer": "https://www.transfermarkt.com/"}
        )

    def _get(self, url: str) -> BeautifulSoup:
        time.sleep(self._delay)
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")

    def get_player_values(self, competition: str) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
          player, position, team, market_value_eur

        market_value_eur is in raw euros (e.g. 50_000_000 for €50m).
        """
        url = TRANSFERMARKT_COMP_URLS.get(competition)
        if url is None:
            raise ValueError(
                f"No Transfermarkt URL for '{competition}'. "
                f"Available: {list(TRANSFERMARKT_COMP_URLS)}"
            )

        records: list[dict] = []
        page = 1
        while True:
            page_url = url if page == 1 else f"{url}/page/{page}"
            logger.info("Transfermarkt page %d: %s", page, page_url)
            soup = self._get(page_url)
            rows = self._parse_table(soup)
            if not rows:
                break
            records.extend(rows)

            # Check for next page link
            next_link = soup.find("li", class_="naechste-seite")
            if not next_link or not next_link.find("a"):
                break
            page += 1

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No Transfermarkt data found for %s", competition)
            return df

        df["market_value_eur"] = df["market_value_raw"].apply(_parse_tm_value)
        return df[["player", "position", "team", "market_value_eur"]].copy()

    @staticmethod
    def _parse_table(soup: BeautifulSoup) -> list[dict]:
        """Extract player rows from a Transfermarkt market-values table."""
        records = []
        table = soup.find("table", class_="items")
        if table is None:
            return records

        for row in table.find_all("tr", class_=["odd", "even"]):
            cells = row.find_all("td")
            if len(cells) < 6:
                continue

            name_tag = row.find("a", class_="spielprofil_tooltip") or row.find(
                "td", class_="hauptlink"
            )
            player_name = name_tag.get_text(strip=True) if name_tag else ""
            if not player_name:
                continue

            # Position: usually the 4th or 5th td depending on layout
            pos_td = None
            for td in cells:
                text = td.get_text(strip=True)
                if text and len(text) < 4 and text.upper() in _POS_LOOKUP:
                    pos_td = text
                    break
            position = pos_td or "MF"

            # Club name (td with class "zentriert" containing img + text fallback)
            club = ""
            for td in cells:
                img = td.find("img", attrs={"class": None})
                if img and img.get("alt") and "flag" not in str(img.get("class", "")):
                    club = img["alt"]
                    break

            # Market value: rightmost td with € symbol
            value_raw = ""
            for td in reversed(cells):
                text = td.get_text(strip=True)
                if "€" in text or text.endswith("m") or text.endswith("k"):
                    value_raw = text
                    break

            records.append(
                {
                    "player": player_name,
                    "position": position,
                    "team": club,
                    "market_value_raw": value_raw,
                }
            )
        return records


# ---------------------------------------------------------------------------
# Sofascore scraper
# ---------------------------------------------------------------------------

SOFASCORE_TOURNAMENT_IDS: dict[str, int] = {
    "world_cup_2026": 16,
    "world_cup_2022": 16,
    "copa_america_2024": 133,
    "euro_2024": 1,
    "bundesliga_2024": 35,
    "premier_league_2024": 17,
}

_SOFASCORE_API = "https://api.sofascore.com/api/v1"


class SofascoreScraper:
    """
    Fetches per-player average match ratings from Sofascore's unofficial API.

    The API is public and widely used by the community, but undocumented.
    It uses JSON responses and requires standard browser-like headers.

    Sofascore weights per-match ratings by the quality of the opponent and
    the player's overall impact, making it a reliable live-form signal.
    """

    def __init__(self, request_delay: float = 2.0):
        self._delay = request_delay
        self._session = requests.Session()
        self._session.headers.update(_API_HEADERS)

    def _get_json(self, url: str) -> dict:
        time.sleep(self._delay)
        resp = self._session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _find_season_id(self, tournament_id: int, year: str) -> int | None:
        """Look up the Sofascore season ID for a given tournament and year."""
        url = f"{_SOFASCORE_API}/unique-tournament/{tournament_id}/seasons"
        try:
            data = self._get_json(url)
            for season in data.get("seasons", []):
                if str(year) in str(season.get("year", "")):
                    return season["id"]
        except Exception as exc:
            logger.warning("Sofascore season lookup failed: %s", exc)
        return None

    def get_player_ratings(
        self,
        competition: str,
        min_appearances: int = 1,
        page_limit: int = 5,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
          player, team, position, sofascore_rating, appearances

        Players with fewer than min_appearances matches are excluded so that
        a single lucky game doesn't inflate a player's quality score.
        """
        tid = SOFASCORE_TOURNAMENT_IDS.get(competition)
        if tid is None:
            raise ValueError(
                f"No Sofascore ID for '{competition}'. "
                f"Available: {list(SOFASCORE_TOURNAMENT_IDS)}"
            )

        year = competition.split("_")[-1]
        sid = self._find_season_id(tid, year)
        if sid is None:
            logger.warning(
                "Could not find Sofascore season for %s. "
                "Tournament may not have started yet.",
                competition,
            )
            return pd.DataFrame(
                columns=["player", "team", "position", "sofascore_rating", "appearances"]
            )

        records: list[dict] = []
        for offset in range(0, page_limit * 100, 100):
            url = (
                f"{_SOFASCORE_API}/unique-tournament/{tid}/season/{sid}"
                f"/statistics/player?limit=100&offset={offset}"
                f"&order=-rating&accumulation=total"
                f"&fields=player.name,player.position,team.name,rating,matchesStarted"
            )
            try:
                data = self._get_json(url)
            except Exception as exc:
                logger.warning("Sofascore request failed at offset %d: %s", offset, exc)
                break

            results = data.get("results", [])
            if not results:
                break

            for entry in results:
                p = entry.get("player", {})
                t = entry.get("team", {})
                stats = entry.get("statistics", entry)  # shape varies by endpoint
                appearances = int(stats.get("matchesStarted", stats.get("appearances", 0)))
                rating = float(stats.get("rating", 0))
                if appearances < min_appearances or rating == 0:
                    continue
                records.append(
                    {
                        "player": p.get("name", ""),
                        "team": t.get("name", ""),
                        "position": p.get("position", "M"),
                        "sofascore_rating": rating,
                        "appearances": appearances,
                    }
                )

            if len(results) < 100:
                break  # last page

        df = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["player", "team", "position", "sofascore_rating", "appearances"]
        )
        logger.info("Sofascore: %d players fetched for %s", len(df), competition)
        return df


# ---------------------------------------------------------------------------
# FBref percentile computer (no extra requests — uses existing merged stats)
# ---------------------------------------------------------------------------


_FBREF_QUALITY_COLS = [
    # offensive output
    ("standard_xG", 1.0),
    ("standard_Ast", 0.8),
    ("passing_KP", 0.8),
    ("shooting_npxG_Sh", 0.9),
    # defensive output
    ("defense_TklW", 0.7),
    ("defense_Int", 0.7),
    ("defense_Clr", 0.5),
    # possession
    ("possession_Take-Ons_Succ%", 0.6),
    ("passing_Total_Cmp%", 0.6),
]


def compute_fbref_percentile(
    fbref_stats: pd.DataFrame, pos_col: str = "standard_Pos"
) -> pd.Series:
    """
    Compute a composite quality score from FBref stats alone.
    Returns a Series indexed by player name, values in [0, 1].
    Each metric is weighted, then percentile-ranked within position group.
    """
    df = fbref_stats.copy()
    if "player" not in df.columns:
        df = df.reset_index()
        if "player" not in df.columns:
            raise ValueError("FBref stats must have a 'player' column.")

    df["_pos_group"] = df[pos_col].astype(str).apply(_position_group) if pos_col in df.columns else "midfielder"

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for col, weight in _FBREF_QUALITY_COLS:
        # Accept partial column name matches
        matched = next((c for c in df.columns if col.lower() in c.lower()), None)
        if matched is None:
            continue
        vals = pd.to_numeric(df[matched], errors="coerce").fillna(0)
        composite += weight * vals
        total_weight += weight

    if total_weight > 0:
        composite /= total_weight

    # Rank within position group then scale to [0, 1]
    df["_composite"] = composite
    result = _minmax_by_group(df, "_composite", "_pos_group")
    result.index = df["player"]
    result.name = "fbref_quality"
    return result


# ---------------------------------------------------------------------------
# Quality score builder (combines all three sources)
# ---------------------------------------------------------------------------


class QualityScoreBuilder:
    """
    Combines Transfermarkt values, FBref percentiles, and Sofascore ratings
    into a single per-player quality_score in [0, 1].

    Weight redistribution
    ---------------------
    When a player has no Sofascore data (e.g. tournament hasn't started),
    their Sofascore weight is split proportionally between the other two
    sources so the final score still sums to 1.

    Sofascore weight scales linearly with appearances up to min_full_apps,
    preventing one lucky match from dominating the signal.

    Parameters
    ----------
    w_transfermarkt : float   weight for market value  (default 0.40)
    w_fbref         : float   weight for FBref composite (default 0.30)
    w_sofascore     : float   max weight for Sofascore  (default 0.30)
    min_full_apps   : int     appearances needed for full Sofascore weight
    name_match_cutoff : float fuzzy name matching threshold [0-1]
    """

    def __init__(
        self,
        w_transfermarkt: float = 0.40,
        w_fbref: float = 0.30,
        w_sofascore: float = 0.30,
        min_full_apps: int = 3,
        name_match_cutoff: float = 0.80,
    ):
        self.w_tm = w_transfermarkt
        self.w_fbr = w_fbref
        self.w_sof = w_sofascore
        self.min_full_apps = min_full_apps
        self.cutoff = name_match_cutoff

        if abs(w_transfermarkt + w_fbref + w_sofascore - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def build(
        self,
        fbref_stats: pd.DataFrame,
        tm_values: pd.DataFrame | None = None,
        sofascore_ratings: pd.DataFrame | None = None,
        pos_col: str = "standard_Pos",
    ) -> pd.DataFrame:
        """
        Build the quality score table.

        Parameters
        ----------
        fbref_stats      : merged FBref DataFrame (from FBrefScraper)
        tm_values        : output of TransfermarktScraper.get_player_values()
        sofascore_ratings: output of SofascoreScraper.get_player_ratings()
        pos_col          : column in fbref_stats with position strings

        Returns
        -------
        DataFrame indexed by player name with columns:
          quality_score, tm_score, fbref_score, sofascore_score,
          market_value_eur, sofascore_rating, appearances, position_group
        """
        if "player" not in fbref_stats.columns:
            fbref_stats = fbref_stats.reset_index()

        players = fbref_stats["player"].tolist()
        result = pd.DataFrame({"player": players})
        result["position_group"] = (
            fbref_stats[pos_col].astype(str).apply(_position_group).values
            if pos_col in fbref_stats.columns
            else "midfielder"
        )

        # -- FBref score (always available) -----------------------------------
        fbref_scores = compute_fbref_percentile(fbref_stats, pos_col)
        result["fbref_score"] = result["player"].map(fbref_scores).fillna(0.5)

        # -- Transfermarkt score ----------------------------------------------
        result["market_value_eur"] = 0.0
        result["tm_score"] = 0.5  # default mid-range when unavailable

        if tm_values is not None and not tm_values.empty:
            tm_names = tm_values["player"].tolist()
            for i, player in enumerate(result["player"]):
                match = _match_player(player, tm_names, cutoff=self.cutoff)
                if match:
                    row = tm_values[tm_values["player"] == match].iloc[0]
                    result.at[i, "market_value_eur"] = row["market_value_eur"]

            # Normalise within position group
            result["tm_score"] = _minmax_by_group(
                result.assign(_pos=result["position_group"]),
                "market_value_eur",
                "_pos",
            )

        # -- Sofascore score (live signal) ------------------------------------
        result["sofascore_rating"] = np.nan
        result["appearances"] = 0
        result["sofascore_score"] = np.nan

        if sofascore_ratings is not None and not sofascore_ratings.empty:
            sof_names = sofascore_ratings["player"].tolist()
            for i, player in enumerate(result["player"]):
                match = _match_player(player, sof_names, cutoff=self.cutoff)
                if match:
                    row = sofascore_ratings[sofascore_ratings["player"] == match].iloc[0]
                    result.at[i, "sofascore_rating"] = row["sofascore_rating"]
                    result.at[i, "appearances"] = row["appearances"]

            # Normalise ratings within position group (Sofascore ~6–8 range)
            has_rating = result["sofascore_rating"].notna()
            if has_rating.any():
                tmp = result[has_rating].copy()
                tmp["_pos"] = tmp["position_group"]
                normed = _minmax_by_group(tmp, "sofascore_rating", "_pos")
                result.loc[has_rating, "sofascore_score"] = normed.values

        # -- Combine with dynamic weights -------------------------------------
        result["quality_score"] = result.apply(
            lambda r: self._combine(r), axis=1
        )

        result = result.set_index("player")
        return result

    def _combine(self, row: pd.Series) -> float:
        """Per-row weight calculation with Sofascore scaling by appearances."""
        apps = row.get("appearances", 0)
        sof_val = row.get("sofascore_score")
        has_sof = pd.notna(sof_val) and apps > 0

        # Scale Sofascore weight by how many matches played
        w_sof_eff = 0.0
        if has_sof:
            scale = min(1.0, apps / self.min_full_apps)
            w_sof_eff = self.w_sof * scale

        # Redistribute remaining Sofascore weight to TM and FBref proportionally
        leftover = self.w_sof - w_sof_eff
        denom = self.w_tm + self.w_fbr
        w_tm_eff = self.w_tm + leftover * (self.w_tm / denom)
        w_fbr_eff = self.w_fbr + leftover * (self.w_fbr / denom)

        tm_val = float(row.get("tm_score", 0.5))
        fbr_val = float(row.get("fbref_score", 0.5))
        sof_val = float(sof_val) if has_sof else 0.0

        score = w_tm_eff * tm_val + w_fbr_eff * fbr_val + w_sof_eff * sof_val
        return float(np.clip(score, 0.0, 1.0))
