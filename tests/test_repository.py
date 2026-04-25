"""Unit tests for the PhotoRepository abstraction.

Only JsonPhotoRepository is tested here — it needs no external services.
QdrantPhotoRepository is covered by integration tests that require a running
Qdrant instance and are skipped when one isn't available.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.storage.repository import (
    JsonPhotoRepository, PhotoRepository, get_repository,
)


@pytest.fixture
def tmp_kb(tmp_path):
    """Write a synthetic photo_index.json and return its path."""
    path = tmp_path / "photo_index.json"
    kb = {
        "metadata": {"created_at": "2026-04-01", "version": "test"},
        "photos": [
            {"id": "u-1", "file_path": "a.jpg", "image_type": "receipt",
             "description": "one", "ocr_text": "ALDI", "entities": []},
            {"id": "u-2", "file_path": "b.jpg", "image_type": "food",
             "description": "two", "ocr_text": "pizza", "entities": []},
            {"id": "u-3", "file_path": "c.jpg", "image_type": "document",
             "description": "three", "ocr_text": "", "entities": []},
        ],
    }
    path.write_text(json.dumps(kb))
    return path


class TestJsonPhotoRepositoryInterface:

    def test_is_photo_repository(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        assert isinstance(repo, PhotoRepository)

    def test_all_photos_returns_every_record(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        photos = repo.all_photos()
        assert len(photos) == 3
        assert {p["id"] for p in photos} == {"u-1", "u-2", "u-3"}

    def test_photo_count_matches_list_len(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        assert repo.photo_count() == len(repo.all_photos())

    def test_get_photo_found(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        photo = repo.get_photo("u-2")
        assert photo is not None
        assert photo["image_type"] == "food"

    def test_get_photo_missing_returns_none(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        assert repo.get_photo("does-not-exist") is None

    def test_metadata_exposed(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        meta = repo.metadata()
        assert meta.get("version") == "test"


class TestJsonPhotoRepositoryCache:

    def test_load_caches_kb(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        first = repo.all_photos()
        # Mutate the file on disk — cached repo should NOT reflect the change
        with open(tmp_kb) as f:
            kb = json.load(f)
        kb["photos"].append(
            {"id": "u-4", "file_path": "d.jpg", "image_type": "other",
             "description": "", "ocr_text": "", "entities": []}
        )
        with open(tmp_kb, "w") as f:
            json.dump(kb, f)
        cached = repo.all_photos()
        assert len(cached) == len(first) == 3

    def test_reload_picks_up_new_writes(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        _ = repo.all_photos()
        with open(tmp_kb) as f:
            kb = json.load(f)
        kb["photos"].append(
            {"id": "u-4", "file_path": "d.jpg", "image_type": "other",
             "description": "", "ocr_text": "", "entities": []}
        )
        with open(tmp_kb, "w") as f:
            json.dump(kb, f)
        repo.reload()
        assert repo.photo_count() == 4


class TestJsonPhotoRepositoryUpsert:

    def test_upsert_new_photo_appends(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        vec = np.zeros(384, dtype=np.float32)
        new_photo = {
            "id": "u-new", "file_path": "n.jpg", "image_type": "receipt",
            "description": "newly added", "ocr_text": "", "entities": [],
        }
        repo.upsert_photo(new_photo, vec)
        with open(tmp_kb) as f:
            kb = json.load(f)
        assert any(p["id"] == "u-new" for p in kb["photos"])

    def test_upsert_duplicate_is_noop(self, tmp_kb):
        repo = JsonPhotoRepository(tmp_kb)
        vec = np.zeros(384, dtype=np.float32)
        before = repo.photo_count()
        repo.upsert_photo(
            {"id": "u-1", "file_path": "a.jpg", "image_type": "receipt",
             "description": "one", "ocr_text": "ALDI", "entities": []},
            vec,
        )
        with open(tmp_kb) as f:
            kb = json.load(f)
        assert len(kb["photos"]) == before


class TestGetRepositoryFactory:

    def test_json_backend_returns_json_repo(self, tmp_kb):
        repo = get_repository(backend="json", kb_path=tmp_kb)
        assert isinstance(repo, JsonPhotoRepository)
        assert repo.photo_count() == 3

    def test_qdrant_backend_falls_back_when_unavailable(self, tmp_kb, monkeypatch):
        """If Qdrant is unreachable, factory must fall back to JSON."""
        # Force QdrantPhotoRepository.__init__ to raise
        import src.storage.repository as repo_mod

        class _FailingQdrant:
            def __init__(self, *a, **kw):
                raise RuntimeError("Qdrant unreachable")

        monkeypatch.setattr(repo_mod, "QdrantPhotoRepository", _FailingQdrant)
        repo = get_repository(backend="qdrant", kb_path=tmp_kb)
        assert isinstance(repo, JsonPhotoRepository)


class TestEmbeddingSearchFallback:

    def test_embedding_search_without_index_returns_empty(self, tmp_kb, monkeypatch):
        """When no npz index exists, embedding_search returns []."""
        repo = JsonPhotoRepository(tmp_kb)

        class _EmptyIndex:
            embeddings = None
            photo_ids = []

            def __init__(self, **kw):
                pass

            def load(self, photos):
                pass

        import src.tools.embedding_index as emb_mod
        monkeypatch.setattr(emb_mod, "EmbeddingIndex", _EmptyIndex)

        results = repo.embedding_search(np.zeros(384, dtype=np.float32), top_k=5)
        assert results == []
