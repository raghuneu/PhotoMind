"""Singleton Qdrant client with retry and health-check."""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
def _connect(url: str) -> QdrantClient:
    """Create and verify a Qdrant connection (retries up to 3x)."""
    client = QdrantClient(url=url, timeout=10)
    # lightweight health probe — raises on failure
    client.get_collections()
    logger.info("Connected to Qdrant at %s", url)
    return client


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a process-wide singleton Qdrant client."""
    settings = get_settings()
    return _connect(settings.qdrant_url)
