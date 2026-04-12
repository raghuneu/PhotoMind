"""
Direct ingestion script — bypasses CrewAI agent overhead to minimize API calls.

Ingestion is batch processing, not agentic reasoning. This script:
  1. Lists all images in photos/
  2. Calls GPT-4o Vision directly (1 API call per photo)
  3. Writes structured records to knowledge_base/photo_index.json

Idempotent: re-running skips already-indexed photos.
"""

import base64
import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image
import pillow_heif
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
pillow_heif.register_heif_opener()

PHOTOS_DIR = Path("./photos")
KB_PATH = Path("./knowledge_base/photo_index.json")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

_ANALYSIS_PROMPT = (
    "Analyze this image in detail. Respond with a JSON object (no markdown, raw JSON only) "
    "with these exact keys:\n"
    '  "image_type": one of [bill, receipt, screenshot, food, scene, document, handwriting, other]\n'
    '  "ocr_text": string with ALL visible text verbatim\n'
    '  "description": 2-3 sentence description of the image\n'
    '  "entities": array of objects with "type" and "value" keys. '
    'Types: amount, date, vendor, food_item, location, person, topic\n'
    '  "confidence": float 0.0-1.0\n\n'
    "For bills and receipts always extract: total amount, vendor name, and date."
)


def load_existing_kb() -> dict:
    if KB_PATH.exists():
        with open(KB_PATH) as f:
            return json.load(f)
    return {"metadata": {}, "photos": []}


def analyze_photo(client: OpenAI, image_path: Path) -> dict:
    img = PIL.Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if GPT-4o wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "image_type": "other",
            "ocr_text": raw[:500],
            "description": "Analysis returned non-JSON response.",
            "entities": [],
            "confidence": 0.3,
        }


def run_direct_ingest():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Check your .env file.")

    client = OpenAI(api_key=api_key)
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)

    kb = load_existing_kb()
    existing_filenames = {p["filename"] for p in kb.get("photos", [])}

    photos = sorted([
        p for p in PHOTOS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    new_photos = [p for p in photos if p.name not in existing_filenames]

    if not new_photos:
        print("All photos already indexed. Nothing to do.")
        return

    print(f"Found {len(new_photos)} new photos to index "
          f"(skipping {len(existing_filenames)} already indexed).\n")

    added = 0
    for i, photo_path in enumerate(new_photos, 1):
        print(f"[{i}/{len(new_photos)}] Analyzing {photo_path.name} ...", end=" ", flush=True)
        try:
            analysis = analyze_photo(client, photo_path)
            record = {
                "id": str(uuid.uuid4()),
                "file_path": str(photo_path),
                "filename": photo_path.name,
                "image_type": analysis.get("image_type", "other"),
                "ocr_text": analysis.get("ocr_text", ""),
                "description": analysis.get("description", ""),
                "entities": analysis.get("entities", []),
                "confidence": float(analysis.get("confidence", 0.5)),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            kb["photos"].append(record)
            added += 1
            print(f"done ({record['image_type']}, confidence={record['confidence']:.2f})")
        except Exception as e:
            print(f"ERROR: {e}")

    kb["metadata"] = {
        "created_at": kb["metadata"].get("created_at", datetime.now(timezone.utc).isoformat()),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_photos": len(kb["photos"]),
    }

    with open(KB_PATH, "w") as f:
        json.dump(kb, f, indent=2)

    print(f"\nDone. {added} new photos indexed. Total: {len(kb['photos'])}.")
    print(f"Knowledge base written to {KB_PATH}")
