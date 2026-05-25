"""
Persistent player → personality_category registry backed by SQLite.

The original notebook imputes unknown players with a random personality
category sampled from the competition-wide distribution:

    all_matches_lineups[i][j][k] = np.random.choice(
        cat_probs['category'], p=cat_probs['count']
    )

This adds noise to every training run because the sample changes each time.
PlayerRegistry replaces that random draw with a deterministic lookup:

  1. Exact name match   → stored category (zero noise)
  2. Normalised match   → accent-stripped / lowercased exact match
  3. Fuzzy match        → difflib closest-match above a configurable cutoff
  4. Probabilistic imputation → same notebook logic, but using DB frequencies
                               so at least the distribution is stable.

Schema
------
players table:
    player_name    TEXT PRIMARY KEY
    category       INTEGER NOT NULL    ← personality cluster ID (1-27 with GK)
    position_group TEXT               ← 'defender' | 'midfielder' | 'forward'
                                         | 'goalkeeper'
    competition    TEXT               ← source competition key
    updated_at     TIMESTAMP
"""

import logging
import random
import sqlite3
import unicodedata
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS players (
    player_name    TEXT PRIMARY KEY,
    category       INTEGER NOT NULL,
    position_group TEXT,
    competition    TEXT,
    updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_UPSERT_SQL = """
INSERT OR REPLACE INTO players (player_name, category, position_group, competition, updated_at)
VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
"""


def _normalise(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace — mirrors quality_scraper.py."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode()
    return " ".join(ascii_name.lower().split())


# ---------------------------------------------------------------------------
# PlayerRegistry
# ---------------------------------------------------------------------------


class PlayerRegistry:
    """
    Persistent SQLite store for player_name → personality_category mappings.

    After clustering runs in the notebook, categories are registered here.
    When building training data from new competitions, unknown players are
    looked up first by exact name, then by fuzzy match, and only fall back
    to probabilistic imputation as a last resort.

    Schema
    ------
    players table:
        player_name   TEXT PRIMARY KEY
        category      INTEGER NOT NULL      ← personality cluster ID (1-27 with GK)
        position_group TEXT               ← 'defender' | 'midfielder' | 'forward' | 'goalkeeper'
        competition   TEXT               ← source competition key
        updated_at    TIMESTAMP

    Parameters
    ----------
    db_path : Path
        Location of the SQLite file. Parent directory is created if absent.
    """

    def __init__(self, db_path: Path = Path("data/player_registry.db")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._name_cache: list[str] | None = None
        self._init_schema()
        logger.info("PlayerRegistry initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def register(
        self,
        player_name: str,
        category: int,
        position_group: str,
        competition: str,
    ) -> None:
        """INSERT OR REPLACE a single player → category mapping."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_UPSERT_SQL, (player_name, int(category), position_group, competition))
            conn.commit()
        self._name_cache = None  # invalidate fuzzy cache
        logger.debug("Registered %s → category %d (%s)", player_name, category, competition)

    def register_bulk(self, df: pd.DataFrame, competition: str) -> None:
        """
        Bulk upsert from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: player_name, category, position_group.
        competition : str
            Competition key stored alongside each record (e.g. 'world_cup_2022').
        """
        required = {"player_name", "category", "position_group"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        records = [
            (row["player_name"], int(row["category"]), row["position_group"], competition)
            for _, row in df.iterrows()
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(_UPSERT_SQL, records)
            conn.commit()
        self._name_cache = None  # invalidate fuzzy cache
        logger.info(
            "Bulk registered %d players from '%s'", len(records), competition
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _load_name_cache(self) -> list[str]:
        """Load all stored player names (used for fuzzy matching). Cached in memory."""
        if self._name_cache is None:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT player_name FROM players;").fetchall()
            self._name_cache = [r[0] for r in rows]
        return self._name_cache

    def lookup(self, player_name: str, cutoff: float = 0.80) -> int | None:
        """
        Return the personality category for a player, or None if not found.

        Resolution order
        ----------------
        1. Exact match on player_name (case-sensitive, as stored).
        2. Normalised exact match (lowercase + accent stripping).
        3. Fuzzy match with difflib.get_close_matches against all stored names.

        Parameters
        ----------
        player_name : str
            Name to look up.
        cutoff : float
            Minimum similarity ratio for fuzzy matching (0–1, default 0.80).

        Returns
        -------
        int or None
        """
        # Step 1: exact match
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT category FROM players WHERE player_name = ?;",
                (player_name,),
            ).fetchone()
        if row is not None:
            logger.debug("Exact match: %s → %d", player_name, row[0])
            return int(row[0])

        # Step 2: normalised exact match
        norm_query = _normalise(player_name)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT player_name, category FROM players;"
            ).fetchall()

        for stored_name, stored_cat in rows:
            if _normalise(stored_name) == norm_query:
                logger.debug(
                    "Normalised match: %s → %s (category %d)",
                    player_name,
                    stored_name,
                    stored_cat,
                )
                return int(stored_cat)

        # Step 3: fuzzy match
        all_names = self._load_name_cache()
        if not all_names:
            return None

        norm_candidates = [_normalise(n) for n in all_names]
        matches = get_close_matches(norm_query, norm_candidates, n=1, cutoff=cutoff)
        if matches:
            idx = norm_candidates.index(matches[0])
            matched_name = all_names[idx]
            # Retrieve category for the matched original name
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT category FROM players WHERE player_name = ?;",
                    (matched_name,),
                ).fetchone()
            if row is not None:
                logger.debug(
                    "Fuzzy match: %s → %s (category %d, similarity >= %.2f)",
                    player_name,
                    matched_name,
                    row[0],
                    cutoff,
                )
                return int(row[0])

        return None

    def impute_category(self, position_group: str) -> int:
        """
        Probabilistic fallback when no match is found for a player.

        Reproduces the original notebook logic deterministically:
        samples a category proportionally to its frequency in the DB
        for the given position_group. Falls back to all categories if
        no data exists for that position group.

        Parameters
        ----------
        position_group : str
            One of 'defender', 'midfielder', 'forward', 'goalkeeper'.

        Returns
        -------
        int
            A category ID sampled from the stored frequency distribution.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT category, COUNT(*) AS cnt
                FROM players
                WHERE position_group = ?
                GROUP BY category;
                """,
                (position_group,),
            ).fetchall()

        if not rows:
            # No data for this position group — fall back to all categories
            logger.debug(
                "No data for position_group '%s'; sampling from all categories.",
                position_group,
            )
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT category, COUNT(*) AS cnt
                    FROM players
                    GROUP BY category;
                    """
                ).fetchall()

        if not rows:
            raise RuntimeError(
                "PlayerRegistry is empty. Register at least one player before imputing."
            )

        categories = [r[0] for r in rows]
        weights = [r[1] for r in rows]
        chosen = random.choices(categories, weights=weights, k=1)[0]
        logger.debug(
            "Imputed category %d for position_group '%s'", chosen, position_group
        )
        return int(chosen)

    def resolve(
        self,
        player_name: str,
        position_group: str,
        cutoff: float = 0.80,
    ) -> tuple[int, str]:
        """
        Convenience method: look up a player and, if needed, impute a category.

        Parameters
        ----------
        player_name : str
            Name to resolve.
        position_group : str
            Used only when imputation is required.
        cutoff : float
            Fuzzy-match cutoff forwarded to lookup().

        Returns
        -------
        (category, source) where source is one of:
            'exact'    – returned by exact or normalised match
            'fuzzy'    – returned by fuzzy match
            'imputed'  – probabilistically sampled from DB frequencies
        """
        # Determine source by checking what lookup would use.
        # We call lookup() once and infer source from a minimal re-check.

        # Step 1: exact
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT category FROM players WHERE player_name = ?;",
                (player_name,),
            ).fetchone()
        if row is not None:
            return int(row[0]), "exact"

        # Step 2: normalised exact
        norm_query = _normalise(player_name)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT player_name, category FROM players;"
            ).fetchall()
        for stored_name, stored_cat in rows:
            if _normalise(stored_name) == norm_query:
                return int(stored_cat), "exact"

        # Step 3: fuzzy
        all_names = self._load_name_cache()
        if all_names:
            norm_candidates = [_normalise(n) for n in all_names]
            matches = get_close_matches(norm_query, norm_candidates, n=1, cutoff=cutoff)
            if matches:
                idx = norm_candidates.index(matches[0])
                matched_name = all_names[idx]
                with sqlite3.connect(self.db_path) as conn:
                    row = conn.execute(
                        "SELECT category FROM players WHERE player_name = ?;",
                        (matched_name,),
                    ).fetchone()
                if row is not None:
                    return int(row[0]), "fuzzy"

        # Step 4: impute
        category = self.impute_category(position_group)
        return category, "imputed"

    # ------------------------------------------------------------------
    # Reporting / introspection
    # ------------------------------------------------------------------

    def all_categories(self) -> pd.DataFrame:
        """
        Return all stored rows as a DataFrame.

        Columns: player_name, category, position_group, competition.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT player_name, category, position_group, competition FROM players;",
                conn,
            )
        return df

    def stats(self) -> dict:
        """
        Return summary statistics about the registry.

        Returns
        -------
        dict with keys:
            total          : int — total number of registered players
            by_competition : dict[str, int] — count per competition key
            by_position    : dict[str, int] — count per position_group
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM players;").fetchone()[0]

            comp_rows = conn.execute(
                "SELECT competition, COUNT(*) FROM players GROUP BY competition;"
            ).fetchall()

            pos_rows = conn.execute(
                "SELECT position_group, COUNT(*) FROM players GROUP BY position_group;"
            ).fetchall()

        return {
            "total": total,
            "by_competition": {r[0]: r[1] for r in comp_rows},
            "by_position": {r[0]: r[1] for r in pos_rows},
        }
