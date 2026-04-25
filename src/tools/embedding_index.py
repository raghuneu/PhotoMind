"""
EmbeddingIndex — sentence-transformer embedding index for PhotoMind.

Builds a dense vector index over photo descriptions and OCR text using
all-MiniLM-L6-v2 (384-dim, runs locally, zero API cost). Supports cosine
similarity search as the 4th retrieval strategy alongside factual, semantic,
and behavioral search.

Design choice — why all-MiniLM-L6-v2:
  - Already in requirements.txt (sentence-transformers) and used by CrewAI
    for agent memory embeddings (see query_crew.py embedder config)
  - 384 dimensions balances quality vs. speed for a corpus of <100 photos
  - Runs on CPU in <1s for our corpus size
  - Outperforms keyword overlap on paraphrase queries ("expensive meal"
    matches "pricey dinner") — the exact weakness noted in
    PhotoKnowledgeBaseTool._semantic_search() docstring

Caching: Embeddings are persisted to knowledge_base/embeddings.npz so they
are only recomputed when the knowledge base changes (detected via photo count
mismatch). This keeps query-time latency to a single matrix multiply.
"""

import os
import numpy as np

# Lazy-load sentence_transformers to avoid import cost on non-embedding paths
_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _photo_text(photo: dict) -> str:
    """Combine a photo's searchable text fields into a single string."""
    parts = []
    desc = photo.get("description", "").strip()
    if desc:
        parts.append(desc)
    ocr = photo.get("ocr_text", "").strip()
    if ocr:
        parts.append(ocr)
    caption = photo.get("caption", "").strip()
    if caption:
        parts.append(caption)
    # Include entity values for factual grounding
    for entity in photo.get("entities", []):
        val = entity.get("value", "").strip()
        etype = entity.get("type", "").strip()
        if val:
            parts.append(f"{etype}: {val}")
    return " | ".join(parts) if parts else photo.get("filename", "unknown photo")


class EmbeddingIndex:
    """Dense vector index over the photo knowledge base.

    Usage:
        idx = EmbeddingIndex(kb_path="./knowledge_base/photo_index.json")
        idx.build()  # or idx.load() if cache exists
        results = idx.search("expensive restaurant meal", top_k=5)
    """

    def __init__(self, kb_path: str = "./knowledge_base/photo_index.json"):
        self.kb_path = kb_path
        self._cache_path = os.path.join(
            os.path.dirname(kb_path), "embeddings.npz"
        )
        self.embeddings: np.ndarray | None = None
        self.photo_ids: list[str] = []
        self.photo_texts: list[str] = []

    def build(self, photos: list[dict] | None = None) -> None:
        """Build embeddings from the knowledge base photos.

        If photos is None, loads from self.kb_path.
        """
        import json

        if photos is None:
            with open(self.kb_path) as f:
                kb = json.load(f)
            photos = kb.get("photos", [])

        if not photos:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.photo_ids = []
            self.photo_texts = []
            return

        self.photo_ids = [p["id"] for p in photos]
        self.photo_texts = [_photo_text(p) for p in photos]

        model = _get_model()
        self.embeddings = model.encode(
            self.photo_texts,
            normalize_embeddings=True,  # unit vectors for cosine sim via dot product
            show_progress_bar=False,
            batch_size=32,
        ).astype(np.float32)

        self._save_cache()

    def _save_cache(self) -> None:
        """Persist embeddings to disk for fast reload."""
        if self.embeddings is not None:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            np.savez_compressed(
                self._cache_path,
                embeddings=self.embeddings,
                photo_ids=np.array(self.photo_ids, dtype=object),
            )

    def load(self, photos: list[dict] | None = None) -> bool:
        """Load cached embeddings. Returns True if cache is valid.

        Rebuilds if cache is missing or photo count has changed.
        """
        import json

        if photos is None:
            try:
                with open(self.kb_path) as f:
                    kb = json.load(f)
                photos = kb.get("photos", [])
            except (FileNotFoundError, json.JSONDecodeError):
                return False

        if not os.path.exists(self._cache_path):
            self.build(photos)
            return True

        try:
            data = np.load(self._cache_path, allow_pickle=True)
            cached_ids = list(data["photo_ids"])
            # Invalidate cache if photo count changed
            if len(cached_ids) != len(photos):
                self.build(photos)
                return True
            self.embeddings = data["embeddings"]
            self.photo_ids = cached_ids
            self.photo_texts = [_photo_text(p) for p in photos]
            return True
        except Exception:
            self.build(photos)
            return True

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index by cosine similarity.

        Returns list of {photo_id, score, text_preview} sorted by score desc.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        model = _get_model()
        query_emb = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Cosine similarity via dot product (embeddings are unit-normalized)
        scores = (self.embeddings @ query_emb.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "photo_id": self.photo_ids[idx],
                "score": round(score, 4),
                "text_preview": self.photo_texts[idx][:200],
            })
        return results
