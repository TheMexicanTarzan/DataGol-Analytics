"""
StatsBomb open-data scraper — aggregates per-match event data into a
per-player stats DataFrame with column names compatible with
PersonalityFeatureBuilder._col().

Supported competitions (StatsBomb free open data only):
  copa_america_2024, world_cup_2022, euro_2024, world_cup_2018

Column ordering is critical: when two columns share a substring fragment,
the more specific column must come FIRST so _col() matches the right one.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import DataCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Competition map  (competition_id, season_id) from StatsBomb open data
# ---------------------------------------------------------------------------

STATSBOMB_COMPETITION_MAP: dict[str, tuple[int, int]] = {
    "copa_america_2024": (223, 282),
    "world_cup_2022":    (43,  106),
    "euro_2024":         (55,  282),
    "world_cup_2018":    (43,  3),
}

# StatsBomb position name → FBref-style code (used by PersonalityFeatureBuilder)
_SB_POS_MAP: dict[str, str] = {
    "Goalkeeper":               "GK",
    "Right Center Back":        "RCB",
    "Left Center Back":         "LCB",
    "Center Back":              "CB",
    "Right Back":               "RB",
    "Left Back":                "LB",
    "Right Wing Back":          "RWB",
    "Left Wing Back":           "LWB",
    "Center Defensive Midfield":"DM",
    "Right Center Midfield":    "CM",
    "Left Center Midfield":     "CM",
    "Center Midfield":          "CM",
    "Right Midfield":           "RM",
    "Left Midfield":            "LM",
    "Center Attacking Midfield":"CAM",
    "Right Attacking Midfield": "RAM",
    "Left Attacking Midfield":  "LAM",
    "Right Wing":               "RW",
    "Left Wing":                "LW",
    "Right Center Forward":     "CF",
    "Left Center Forward":      "CF",
    "Center Forward":           "CF",
    "Secondary Striker":        "SS",
}


class StatsBombScraper:
    """
    Aggregates StatsBomb open-data events into per-player stats.

    The output DataFrame uses column names that PersonalityFeatureBuilder._col()
    can locate via substring search.  Column ORDER matters — shorter fragments
    must come before any column whose name contains them as a substring.

    Example mapping (StatsBomb → column used by _col()):
      Pass events                    → pass_att, pass_cmp, crs, kp, ast
      Shot events                    → sh (shots), gls, xg, npxg_sh
      Dribble events                 → dribble_att, dribble_succ
      50/50 events                   → aerialwon, aeriallost
      Clearance / Block / Intercept  → clr, block_blocks, block_sh, int
      Duel (Tackle) events           → tkl, tklw
      Foul events                    → fls, crdy, crdr
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        ttl_hours: int = 720,
    ):
        self._cache = DataCache(cache_dir=cache_dir, ttl_hours=ttl_hours)

    def is_supported(self, competition: str) -> bool:
        return competition in STATSBOMB_COMPETITION_MAP

    def get_merged_player_stats(self, competition: str) -> pd.DataFrame:
        """
        Fetch and aggregate all events for the competition into a per-player
        stats DataFrame.  Results are cached to disk (ttl_hours=720 by default
        since historical open data never changes).
        """
        cache_key = f"{competition}_statsbomb_player_stats"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("StatsBomb: loaded %s from cache (%d players).", competition, len(cached))
            return cached

        try:
            from statsbombpy import sb
        except ImportError as exc:
            raise ImportError(
                "statsbombpy is not installed. Run: pip install statsbombpy"
            ) from exc

        comp_id, season_id = STATSBOMB_COMPETITION_MAP[competition]
        logger.info(
            "StatsBomb: fetching matches for competition_id=%d season_id=%d",
            comp_id, season_id,
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = sb.matches(competition_id=comp_id, season_id=season_id)

        if matches.empty:
            raise ValueError(f"StatsBomb: no matches found for {competition}")

        logger.info("StatsBomb: %d matches found, fetching events…", len(matches))

        all_events: list[pd.DataFrame] = []
        for _, match in matches.iterrows():
            match_id = int(match["match_id"])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ev = sb.events(match_id=match_id)
                all_events.append(ev)
                logger.debug("StatsBomb: fetched events for match %d", match_id)
            except Exception as exc:
                logger.warning("StatsBomb: skipping match %d — %s", match_id, exc)

        if not all_events:
            raise ValueError(f"StatsBomb: could not fetch any events for {competition}")

        events = pd.concat(all_events, ignore_index=True)
        logger.info(
            "StatsBomb: %d total events across %d matches, aggregating…",
            len(events), len(all_events),
        )

        df = self._aggregate_events(events)
        self._cache.set(cache_key, df)
        logger.info("StatsBomb: aggregated stats for %d players.", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate all StatsBomb events into one row per player.

        Column order is chosen carefully so PersonalityFeatureBuilder._col()
        finds the right column when two columns share a common fragment.
        Specifically:
          "tkl"  must precede "tklw"    (both contain "tkl")
          "sh"   must precede "short_*" and "shot_*"  (all contain "sh")
          "xg"   must precede "npxg_sh" ("xg" is in "npxg")
          "ast"  must precede "dribbled_past" ("ast" is in "past")
          "ck"   must precede "block_blocks"  ("ck" is in "block")
          "crs"  must precede nothing problematic, but kept early
        """
        if "player" not in events.columns:
            return pd.DataFrame()

        events = events.dropna(subset=["player"])
        type_col = events["type"] if "type" in events.columns else pd.Series("", index=events.index)

        rows: list[dict] = []

        for player_name, p_ev in events.groupby("player", sort=False):
            # ── Position ────────────────────────────────────────────────
            pos_series = p_ev.get("position", pd.Series(dtype=str)).dropna()
            raw_pos = str(pos_series.mode().iloc[0]) if not pos_series.empty else "Unknown"
            pos = _SB_POS_MAP.get(raw_pos, "MF")

            # ── Minutes (approximate: last event minute) ─────────────────
            minutes = float(p_ev["minute"].max()) if "minute" in p_ev.columns else 90.0

            p_type = p_ev["type"] if "type" in p_ev.columns else pd.Series("", index=p_ev.index)

            # ── Passes ──────────────────────────────────────────────────
            pass_ev    = p_ev[p_type == "Pass"]
            pass_att   = len(pass_ev)

            if "pass_outcome" in pass_ev.columns:
                pass_out = pass_ev["pass_outcome"]
                pass_cmp = int(pass_out.isna().sum())
            else:
                pass_out = pd.Series(dtype=str)
                pass_cmp = pass_att  # assume all successful if column missing

            crosses    = _flag_sum(pass_ev, "pass_cross")
            key_passes = _flag_sum(pass_ev, "pass_shot_assist")
            assists    = _flag_sum(pass_ev, "pass_goal_assist")

            if "pass_length" in pass_ev.columns:
                pass_len  = pass_ev["pass_length"].fillna(0)
                short_att = int((pass_len < 18).sum())
                medium_att= int(((pass_len >= 18) & (pass_len < 36)).sum())
                long_att  = int((pass_len >= 36).sum())
                if len(pass_len) == len(pass_out):
                    short_cmp = int(((pass_len < 18) & pass_out.isna()).sum())
                else:
                    short_cmp = short_att
            else:
                pass_len = pd.Series(dtype=float)
                short_att = medium_att = long_att = short_cmp = 0

            if "pass_type" in pass_ev.columns:
                pass_type  = pass_ev["pass_type"].fillna("").str.lower()
                corners    = int(pass_type.str.contains("corner", na=False).sum())
                throw_ins  = int(pass_type.str.contains("throw",  na=False).sum())
            else:
                corners = throw_ins = 0
            dead_ball = int((pass_len >= 36).sum()) if len(pass_len) > 0 else 0

            # ── Shots ───────────────────────────────────────────────────
            shot_ev   = p_ev[p_type == "Shot"]
            shots     = len(shot_ev)

            shot_out  = shot_ev.get("shot_outcome", pd.Series(dtype=str)).fillna("") \
                        if "shot_outcome" in shot_ev.columns else pd.Series(dtype=str)
            goals     = int((shot_out == "Goal").sum())

            if "shot_statsbomb_xg" in shot_ev.columns:
                total_xg  = float(shot_ev["shot_statsbomb_xg"].fillna(0).sum())
            else:
                total_xg  = 0.0
            npxg_sh   = total_xg / shots if shots > 0 else 0.0

            aerial_shots = int(
                shot_ev.get("shot_type", pd.Series(dtype=str)).fillna("").str.lower().eq("header").sum()
            ) if "shot_type" in shot_ev.columns else 0

            # ── Dribbles ────────────────────────────────────────────────
            drib_ev   = p_ev[p_type == "Dribble"]
            drib_att  = len(drib_ev)
            drib_out  = drib_ev.get("dribble_outcome", pd.Series(dtype=str)).fillna("") \
                        if "dribble_outcome" in drib_ev.columns else pd.Series(dtype=str)
            drib_succ = int((drib_out == "Complete").sum())

            # ── 50/50 (aerial duels) ─────────────────────────────────────
            fifty_ev  = p_ev[p_type == "50/50"]
            fifty_out = fifty_ev.get("50_50_outcome", pd.Series(dtype=str)).fillna("") \
                        if "50_50_outcome" in fifty_ev.columns else pd.Series(dtype=str)
            aerial_won  = int(fifty_out.isin(["Success In Play", "Success Out"]).sum())
            aerial_lost = len(fifty_ev) - aerial_won

            # ── Clearances ───────────────────────────────────────────────
            clear_ev = p_ev[p_type == "Clearance"]
            clr      = len(clear_ev)

            # ── Blocks ───────────────────────────────────────────────────
            block_ev     = p_ev[p_type == "Block"]
            blocks_total = len(block_ev)
            blocks_sh    = _flag_sum(block_ev, "block_save_block")
            blocks_pass  = _flag_sum(block_ev, "block_deflection")

            # ── Interceptions ────────────────────────────────────────────
            inter_ev      = p_ev[p_type == "Interception"]
            interceptions = len(inter_ev)

            # ── Duels / tackles ──────────────────────────────────────────
            duel_ev      = p_ev[p_type == "Duel"]
            if "duel_type" in duel_ev.columns:
                tackle_ev = duel_ev[duel_ev["duel_type"].fillna("") == "Tackle"]
            else:
                tackle_ev = pd.DataFrame()
            tkl_total    = len(tackle_ev)
            tkl_out      = tackle_ev.get("duel_outcome", pd.Series(dtype=str)).fillna("") \
                           if "duel_outcome" in tackle_ev.columns else pd.Series(dtype=str)
            tkl_won      = int(tkl_out.isin(["Won", "Success In Play", "Success Out"]).sum())

            # ── Fouls / discipline ───────────────────────────────────────
            foul_ev      = p_ev[p_type == "Foul Committed"]
            fouls        = len(foul_ev)
            bad          = foul_ev.get("bad_behaviour_card", pd.Series(dtype=str)).fillna("") \
                           if "bad_behaviour_card" in foul_ev.columns else pd.Series(dtype=str)
            yellow_cards = int((bad == "Yellow Card").sum())
            red_cards    = int(bad.isin(["Red Card", "Second Yellow"]).sum())

            # ── Assemble row (column order is significant for _col()) ─────
            rows.append({
                "Player": player_name,
                "Pos":    pos,
                "Min":    minutes,
                # Single-fragment lookups — must come before any column whose
                # name CONTAINS the fragment as a substring.
                "tkl":          tkl_total,   # before "tklw"
                "tklw":         tkl_won,
                "sh":           shots,        # before "short_*", "shot_*", "block_sh", "npxg_sh"
                "gls":          goals,
                "xg":           total_xg,    # before "npxg_sh" (contains "xg")
                "ast":          assists,      # before "dribbled_past" (contains "ast")
                "int":          interceptions,
                "kp":           key_passes,
                "crs":          crosses,
                "clr":          clr,
                "fls":          fouls,
                "crdy":         yellow_cards,
                "crdr":         red_cards,
                "ck":           corners,      # before "block_blocks" (contains "ck")
                # Columns whose substrings are already handled above
                "dribbled_past": 0,
                "aerialwon":    aerial_won,
                "aeriallost":   aerial_lost,
                "block_blocks": blocks_total,
                "block_sh":     blocks_sh,
                "block_pass":   blocks_pass,
                "pass_att":     pass_att,
                "pass_cmp":     pass_cmp,
                "total_cmp":    pass_cmp,
                "short_att":    short_att,
                "short_cmp":    short_cmp,
                "medium_att":   medium_att,
                "long_att":     long_att,
                "corner":       corners,
                "ti":           throw_ins,
                "dead":         dead_ball,
                "npxg_sh":      npxg_sh,
                "shot_aerial":  aerial_shots,
                "dribble_att":  drib_att,
                "dribble_succ": drib_succ,
            })

        return pd.DataFrame(rows)


def _flag_sum(df: pd.DataFrame, col: str) -> int:
    """Sum a boolean/flag column; returns 0 if the column doesn't exist."""
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())
