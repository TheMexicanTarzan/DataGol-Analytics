"""Disk-based parquet cache for scraped data with configurable TTL."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")


class DataCache:
    """
    Parquet-based disk cache. Designed for tournament scraping where data
    changes frequently during match days but is static between them.
    Use ttl_hours=6 during the tournament, ttl_hours=720 for historical data.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, ttl_hours: int = 6):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(" ", "_")
        return self.cache_dir / f"{safe}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._path(key)
        if not path.exists():
            return None
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age > self.ttl:
            logger.info("Cache expired for '%s' (age: %s)", key, age)
            return None
        logger.info("Cache hit: '%s'", key)
        return pd.read_parquet(path)

    def set(self, key: str, df: pd.DataFrame) -> None:
        path = self._path(key)
        df.to_parquet(path)
        logger.info("Cached '%s' (%d rows)", key, len(df))

    def invalidate(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.info("Invalidated cache for '%s'", key)

    def invalidate_competition(self, competition: str) -> None:
        for path in self.cache_dir.glob(f"*{competition}*"):
            path.unlink()
            logger.info("Invalidated: %s", path.name)
