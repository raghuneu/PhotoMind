"""
migrate_to_qdrant.py — One-shot migration from JSON KB to Qdrant.

Reads every photo from knowledge_base/photo_index.json, encodes it with
all-MiniLM-L6-v2, and upserts into the Qdrant "photos" collection.
Idempotent: re-running upserts the same IDs (no duplicates).

Usage:
    python -m scripts.migrate_to_qdrant
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.embedding_index import _get_model, _photo_text
from src.storage.repository import QdrantPhotoRepository


def main() -> None:
    kb_path = PROJECT_ROOT / "knowledge_base" / "photo_index.json"
    with open(kb_path) as f:
        kb = json.load(f)

    photos = kb.get("photos", [])
    if not photos:
        print("No photos in knowledge base. Nothing to migrate.")
        return

    print(f"Loading embedding model...")
    model = _get_model()

    print(f"Encoding {len(photos)} photos...")
    texts = [_photo_text(p) for p in photos]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)

    print("Connecting to Qdrant...")
    repo = QdrantPhotoRepository(collection="photos")

    print(f"Upserting {len(photos)} points...")
    for i, (photo, vec) in enumerate(zip(photos, embeddings)):
        repo.upsert_photo(photo, vec)
        if (i + 1) % 10 == 0 or i == len(photos) - 1:
            print(f"  [{i + 1}/{len(photos)}]")

    count = repo.photo_count()
    print(f"\nDone. Qdrant collection 'photos' now has {count} points.")


if __name__ == "__main__":
    main()
