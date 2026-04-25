"""Storage layer — repository pattern for photo knowledge base."""

from src.storage.repository import (
    PhotoRepository,
    JsonPhotoRepository,
    QdrantPhotoRepository,
    get_repository,
)

__all__ = [
    "PhotoRepository",
    "JsonPhotoRepository",
    "QdrantPhotoRepository",
    "get_repository",
]
