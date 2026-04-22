"""
PhotoRepository — abstract storage layer for the photo knowledge base.

Provides three implementations:
  - JsonPhotoRepository  : reads from the flat JSON file (backward compat, RL training)
  - QdrantPhotoRepository: reads/writes Qdrant vector DB for live queries
  - get_repository()     : factory that returns the configured backend with fallback

The repository exposes a minimal interface consumed by PhotoKnowledgeBaseTool
and api/server.py — all search strategies can work against either backend.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Photo dict type alias (same keys as JSON records) ──────────────────
PhotoRecord = dict[str, Any]


# ── Abstract base ──────────────────────────────────────────────────────
class PhotoRepository(ABC):
    """Minimal interface every storage backend must satisfy."""

    @abstractmethod
    def all_photos(self) -> list[PhotoRecord]:
        """Return every photo record (used by behavioral / factual search)."""

    @abstractmethod
    def get_photo(self, photo_id: str) -> PhotoRecord | None:
        """Lookup a single photo by UUID."""

    @abstractmethod
    def photo_count(self) -> int:
        """Total number of indexed photos."""

    @abstractmethod
    def embedding_search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Return top_k results by vector similarity.

        Each result dict has: photo_id, score, and the full PhotoRecord under 'photo'.
        """

    @abstractmethod
    def upsert_photo(self, photo: PhotoRecord, vector: np.ndarray) -> None:
        """Insert or update a single photo (used by ingestion / migration)."""

    @abstractmethod
    def metadata(self) -> dict:
        """Return KB-level metadata (created_at, last_updated, etc.)."""


# ── JSON flat-file implementation ──────────────────────────────────────
class JsonPhotoRepository(PhotoRepository):
    """Reads the legacy photo_index.json — zero external dependencies."""

    def __init__(self, kb_path: str | Path):
        self._kb_path = Path(kb_path)
        self._kb: dict | None = None

    def _load(self) -> dict:
        if self._kb is None:
            with open(self._kb_path) as f:
                self._kb = json.load(f)
        return self._kb

    def reload(self) -> None:
        """Force re-read from disk (useful after ingestion)."""
        self._kb = None

    # -- interface --

    def all_photos(self) -> list[PhotoRecord]:
        return self._load().get("photos", [])

    def get_photo(self, photo_id: str) -> PhotoRecord | None:
        for p in self.all_photos():
            if p["id"] == photo_id:
                return p
        return None

    def photo_count(self) -> int:
        return len(self.all_photos())

    def embedding_search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        """Brute-force cosine search over cached npz embeddings (fallback)."""
        from src.tools.embedding_index import EmbeddingIndex

        idx = EmbeddingIndex(kb_path=str(self._kb_path))
        idx.load(photos=self.all_photos())
        if idx.embeddings is None or len(idx.embeddings) == 0:
            return []

        scores = (idx.embeddings @ query_vector.reshape(-1, 1)).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        photo_map = {p["id"]: p for p in self.all_photos()}
        results = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue
            pid = idx.photo_ids[i]
            results.append({
                "photo_id": pid,
                "score": round(score, 4),
                "photo": photo_map.get(pid, {}),
            })
        return results

    def upsert_photo(self, photo: PhotoRecord, vector: np.ndarray) -> None:
        """Append to JSON (vector ignored — stored in npz by EmbeddingIndex)."""
        kb = self._load()
        existing = {p["id"] for p in kb.get("photos", [])}
        if photo["id"] not in existing:
            kb.setdefault("photos", []).append(photo)
            with open(self._kb_path, "w") as f:
                json.dump(kb, f, indent=2)

    def metadata(self) -> dict:
        return self._load().get("metadata", {})


# ── Qdrant implementation ──────────────────────────────────────────────
class QdrantPhotoRepository(PhotoRepository):
    """Vector-first storage backed by Qdrant.

    Each Qdrant point stores:
      - id   : UUID string (same as JSON photo id)
      - vector: 384-dim all-MiniLM-L6-v2 embedding
      - payload: full PhotoRecord dict (description, entities, ocr_text, …)

    The collection is auto-created on first upsert if it doesn't exist.
    """

    VECTOR_DIM = 384  # all-MiniLM-L6-v2

    def __init__(self, collection: str = "photos"):
        from src.storage.qdrant_client import get_qdrant_client
        self._client = get_qdrant_client()
        self._collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if missing."""
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)

    # -- interface --

    def all_photos(self) -> list[PhotoRecord]:
        """Scroll all points and return their payloads."""
        records: list[PhotoRecord] = []
        offset = None
        while True:
            result = self._client.scroll(
                collection_name=self._collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for pt in points:
                records.append(pt.payload)
            if next_offset is None:
                break
            offset = next_offset
        return records

    def get_photo(self, photo_id: str) -> PhotoRecord | None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        hits = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=photo_id))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )[0]
        return hits[0].payload if hits else None

    def photo_count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def embedding_search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        results = []
        for hit in hits:
            results.append({
                "photo_id": hit.payload.get("id", ""),
                "score": round(hit.score, 4),
                "photo": hit.payload,
            })
        return results

    def upsert_photo(self, photo: PhotoRecord, vector: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=photo["id"],
                    vector=vector.tolist(),
                    payload=photo,
                )
            ],
        )

    def metadata(self) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "backend": "qdrant",
            "collection": self._collection,
            "total_photos": info.points_count or 0,
        }


# ── Factory ────────────────────────────────────────────────────────────
def get_repository(
    backend: str | None = None,
    kb_path: str | Path | None = None,
    collection: str | None = None,
) -> PhotoRepository:
    """Return the configured PhotoRepository.

    Falls back to JsonPhotoRepository if Qdrant is unavailable.
    Use ``collection`` to override the Qdrant collection name (e.g.
    per-user scoping: ``photos_alice``).
    """
    from src.config import get_settings

    settings = get_settings()
    backend = backend or settings.repository_backend
    kb_path = kb_path or settings.knowledge_base_path
    collection = collection or settings.qdrant_collection

    if backend == "qdrant":
        try:
            repo = QdrantPhotoRepository(collection=collection)
            logger.info("Using Qdrant repository (collection=%s)", collection)
            return repo
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s), falling back to JSON", exc)
            return JsonPhotoRepository(kb_path)
    else:
        return JsonPhotoRepository(kb_path)
