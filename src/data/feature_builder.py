"""
Translates FBref aggregated stats into the proportion-based personality
features used by the DataGol clustering pipeline.

StatsBomb provides event-level data (every touch).
FBref provides per-player aggregated totals per tournament.
The same proportions can be computed from either source.

Features that require coordinate data (e.g. pass height, pass trajectory)
and have no FBref equivalent are dropped; the remaining features cover the
core personality signal.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FBref column name fragments (prefix-less, after _merge_stat_tables)
# ---------------------------------------------------------------------------
# After merging, columns look like "defense_TklW", "shooting_npxG_Sh", etc.
# The helpers below resolve these regardless of exact table/column naming
# differences between soccerdata versions and direct HTML scraping.


def _col(df: pd.DataFrame, *fragments: str, default: float = 0.0) -> pd.Series:
    """
    Find the first column whose name contains ALL of the given fragments.
    Returns a zero-filled Series if not found.
    """
    for col in df.columns:
        col_l = col.lower()
        if all(f.lower() in col_l for f in fragments):
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    logger.debug("Column with fragments %s not found; defaulting to %s", fragments, default)
    return pd.Series(default, index=df.index)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan).fillna(np.nan).where(den > 0, np.nan)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


class PersonalityFeatureBuilder:
    """
    Converts a merged FBref player stats DataFrame into the feature matrix
    expected by the DataGol personality clustering pipeline.

    Input  : merged DataFrame from FBrefScraper.get_merged_player_stats()
    Output : DataFrame with one row per player, columns matching DataGol features

    Dropped features (no FBref equivalent):
      - proportion_dribble_nutmeg / overrun (coordinate/event-level only)
      - mode_pass_height                    (coordinate-level only)
      - mode_pass_body_part                 (not tracked in FBref)
      - proportion_shot_first_time          (not tracked in FBref)
      - proportion_shot_open_goal           (not tracked in FBref)
      - mode_shot_body_part / technique / type  (not tracked in FBref)
    """

    MIN_MINUTES = 60  # Minimum minutes played to include a player

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point. Returns the feature matrix indexed by player name."""
        df = df.copy()
        df = self._filter_players(df)

        features = pd.DataFrame(index=df.index)
        features.index = df["player"] if "player" in df.columns else df.index

        # -- Aerial / 50-50 ---------------------------------------------------
        aerial_won = _col(df, "won", "aerial") + _col(df, "aerialwon")
        aerial_lost = _col(df, "lost", "aerial") + _col(df, "aeriallost")
        features["proportion_50_50_won"] = _safe_div(
            aerial_won, aerial_won + aerial_lost
        ).fillna(0)

        # -- Blocks -----------------------------------------------------------
        blocks_sh = _col(df, "block", "sh")
        blocks_pass = _col(df, "block", "pass")
        blocks_total = _col(df, "block", "blocks") + blocks_sh + blocks_pass
        features["proportion_save_block"] = _safe_div(blocks_sh, blocks_total).fillna(0)
        features["proportion_block_deflection"] = _safe_div(blocks_pass, blocks_total).fillna(0)

        # -- Clearances -------------------------------------------------------
        clearances = _col(df, "clr")
        features["proportion_clearance_aerial_won"] = _safe_div(
            aerial_won, clearances + aerial_won + aerial_lost
        ).fillna(0)

        # -- Dribbles ---------------------------------------------------------
        drib_att = _col(df, "take", "att") + _col(df, "dribble", "att")
        drib_succ = _col(df, "take", "succ") + _col(df, "dribble", "succ")
        features["proportion_dribble_complete"] = _safe_div(drib_succ, drib_att).fillna(0)

        # -- Duels ------------------------------------------------------------
        tkl_won = _col(df, "tklw")
        tkl_total = _col(df, "tkl") + _col(df, "dribbled_past")
        features["proportion_duel_won"] = _safe_div(tkl_won, tkl_total).fillna(0)

        duel_aerial_flag = (aerial_won > aerial_lost).astype(int)
        features["mode_duel_type_aerial_lost"] = (1 - duel_aerial_flag)
        features["mode_duel_type_tackle"] = duel_aerial_flag

        # -- Fouls / discipline -----------------------------------------------
        fouls = _col(df, "fls")
        cards = _col(df, "crdy") + _col(df, "crdr")
        features["proportion_foul_dangerous_play"] = _safe_div(cards, fouls + cards).fillna(0)

        # -- Interceptions ----------------------------------------------------
        ints = _col(df, "int")
        def_actions = tkl_total + ints + clearances
        features["proportion_interception_won"] = _safe_div(ints, def_actions).fillna(0)

        # -- Passing ----------------------------------------------------------
        passes_total = _col(df, "pass", "att") + _col(df, "pass", "cmp")
        crosses = _col(df, "crs")
        key_passes = _col(df, "kp")
        assists = _col(df, "ast")

        features["proportion_pass_cross"] = _safe_div(crosses, passes_total).fillna(0)
        features["proportion_pass_shot_assist"] = _safe_div(key_passes, passes_total).fillna(0)
        features["proportion_pass_goal_assist"] = _safe_div(assists, passes_total).fillna(0)

        # Pass completion under pressure (proxy: short pass completion)
        short_cmp = _col(df, "short", "cmp")
        short_att = _col(df, "short", "att")
        features["proportion_pass_under_pressure_complete"] = _safe_div(
            short_cmp, short_att
        ).fillna(0)

        # Cross / assist completion: use overall pass completion as proxy
        pass_cmp = _col(df, "total", "cmp") + short_cmp
        features["proportion_pass_cross_complete"] = features["proportion_pass_cross"]
        features["proportion_pass_goal_assist_complete"] = features["proportion_pass_goal_assist"]
        features["proportion_pass_shot_assist_complete"] = features["proportion_pass_shot_assist"]

        # Mode pass length: classify by dominant pass type (short / medium / long)
        short_att = _col(df, "short", "att")
        med_att = _col(df, "medium", "att")
        long_att = _col(df, "long", "att")
        features["mode_pass_length_short"] = (
            (short_att >= med_att) & (short_att >= long_att)
        ).astype(int)
        features["mode_pass_length_medium"] = (
            (med_att > short_att) & (med_att >= long_att)
        ).astype(int)
        features["mode_pass_length_long"] = (
            (long_att > short_att) & (long_att > med_att)
        ).astype(int)

        # Mode pass type: corner / throw-in / interception / recovery
        corners = _col(df, "corner") + _col(df, "ck")
        throw_ins = _col(df, "ti")
        dead_ball = _col(df, "dead")
        features["mode_pass_type_corner"] = (corners > 0).astype(int)
        features["mode_pass_type_throw_in"] = (throw_ins > 0).astype(int)
        features["mode_pass_type_recovery"] = (dead_ball > corners + throw_ins).astype(int)
        features["mode_pass_type_interception"] = (ints > 0).astype(int)

        # -- Shooting ---------------------------------------------------------
        shots = _col(df, "sh")
        goals = _col(df, "gls")
        npxg_per_sh = _col(df, "npxg", "sh")
        xg = _col(df, "xg")

        features["proportion_goal"] = _safe_div(goals, shots).fillna(0)
        features["mean_shot_statsbomb_xg"] = npxg_per_sh.where(
            npxg_per_sh > 0, _safe_div(xg, shots)
        ).fillna(0)

        # Aerial shot proxy: shots / (shots + header attempts)
        aerial_shots = _col(df, "shot", "aerial")
        features["proportion_shot_aerial_won"] = _safe_div(aerial_shots, shots).fillna(0)

        features["proportion_goal_under_pressure"] = features["proportion_goal"] * (
            1 - features["proportion_50_50_won"]
        )

        # -- Position (one-hot, sourced from the player's dominant position) --
        features = self._add_position_features(df, features)

        # -- Normalise --------------------------------------------------------
        numeric_cols = features.select_dtypes(include="number").columns
        features[numeric_cols] = features[numeric_cols].clip(0, 1)

        features.index.name = "player"
        return features

    # ------------------------------------------------------------------

    def _filter_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop goalkeepers and players with too few minutes."""
        min_col = next(
            (c for c in df.columns if "min" in c.lower() and "90" not in c.lower()), None
        )
        pos_col = next((c for c in df.columns if "pos" in c.lower()), None)

        if pos_col:
            df = df[~df[pos_col].astype(str).str.upper().str.startswith("GK")]

        if min_col:
            df = df[pd.to_numeric(df[min_col], errors="coerce").fillna(0) >= self.MIN_MINUTES]

        return df.reset_index(drop=True)

    def _add_position_features(
        self, df: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """One-hot encode player positions to match the DataGol clustering input."""
        POSITION_MAP = {
            "CB": "center_back",
            "RB": "right_back",
            "LB": "left_back",
            "RWB": "right_wing_back",
            "LWB": "left_wing_back",
            "RCB": "right_center_back",
            "LCB": "left_center_back",
            "DM": "center_defensive_midfield",
            "CM": "center_midfield",
            "RM": "right_midfield",
            "LM": "left_midfield",
            "CAM": "center_attacking_midfield",
            "RAM": "right_attacking_midfield",
            "LAM": "left_attacking_midfield",
            "RCM": "right_center_midfield",
            "LCM": "left_center_midfield",
            "RDM": "right_defensive_midfield",
            "LDM": "left_defensive_midfield",
            "RW": "right_wing",
            "LW": "left_wing",
            "CF": "center_forward",
            "RCF": "right_center_forward",
            "LCF": "left_center_forward",
            "ST": "center_forward",
            "SS": "center_attacking_midfield",
            "FW": "center_forward",
            "MF": "center_midfield",
            "DF": "center_back",
        }

        pos_col = next((c for c in df.columns if "pos" in c.lower()), None)
        all_positions = sorted(set(POSITION_MAP.values()))
        for pos in all_positions:
            features[f"mode_position_{pos}"] = 0

        if pos_col is None:
            return features

        raw_pos = df[pos_col].astype(str).str.strip().str.upper()
        # FBref sometimes lists multiple positions like "DF,MF" — take the first
        primary = raw_pos.str.split(",").str[0].str.strip()

        for idx, pos_str in zip(features.index, primary):
            mapped = POSITION_MAP.get(pos_str)
            if mapped:
                features.loc[idx, f"mode_position_{mapped}"] = 1

        return features
