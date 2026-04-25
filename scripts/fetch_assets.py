"""Fetch photos + photo_index.json from a private Cloudflare R2 bucket.

This runs as a pre-start step in the Docker container (see Dockerfile CMD).
It downloads a tarball of personal photos and the knowledge-base JSON from
R2 so the public Docker image does not have to bake them in.

Behavior:
  - If any required R2 env var is missing, logs a message and exits 0
    (local dev path — existing local files are used).
  - If the expected assets are already present on disk, skips the download
    (idempotent restarts).
  - Otherwise downloads ``R2_ASSETS_KEY`` (default ``photomind-photos.tar.gz``)
    from ``R2_BUCKET`` and extracts it at the project root.

Expected tarball layout (matches ``tar -czf ... photos knowledge_base/photo_index.json``):
    photos/<...>.jpg
    knowledge_base/photo_index.json
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTOS_DIR = PROJECT_ROOT / "photos"
KB_JSON = PROJECT_ROOT / "knowledge_base" / "photo_index.json"
DOWNLOAD_PATH = Path("/tmp/photomind-assets.tar.gz")


def _log(msg: str) -> None:
    print(f"[fetch_assets] {msg}", flush=True)


def _already_present() -> bool:
    """Return True if photos + photo_index.json are already on disk."""
    if not KB_JSON.exists():
        return False
    if not PHOTOS_DIR.is_dir():
        return False
    # Require at least one photo file present.
    try:
        next(p for p in PHOTOS_DIR.iterdir() if p.is_file())
    except StopIteration:
        return False
    return True


def main() -> int:
    endpoint = os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")
    key = os.getenv("R2_ASSETS_KEY", "photomind-photos.tar.gz")

    if not all([endpoint, access_key, secret_key, bucket]):
        _log("R2 env vars not set — skipping download (using local files if present)")
        return 0

    if _already_present():
        _log(f"Assets already present at {PHOTOS_DIR} and {KB_JSON} — skipping download")
        return 0

    try:
        import boto3  # lazy import so local dev without boto3 still works
        from botocore.config import Config
    except ImportError:
        _log("ERROR: boto3 not installed but R2 vars are set — run `pip install boto3`")
        return 1

    _log(f"Downloading s3://{bucket}/{key} from {endpoint}")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )

    try:
        DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(DOWNLOAD_PATH))
        size_mb = DOWNLOAD_PATH.stat().st_size / (1024 * 1024)
        _log(f"Downloaded {DOWNLOAD_PATH} ({size_mb:.1f} MB)")
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to download tarball: {exc}")
        return 1

    try:
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(DOWNLOAD_PATH, "r:gz") as tf:
            tf.extractall(PROJECT_ROOT)
        _log(f"Extracted to {PROJECT_ROOT}")
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to extract tarball: {exc}")
        return 1

    if not _already_present():
        _log("ERROR: extraction completed but expected files still missing")
        return 1

    photo_count = sum(1 for p in PHOTOS_DIR.iterdir() if p.is_file())
    _log(f"OK: {photo_count} photos + photo_index.json ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
