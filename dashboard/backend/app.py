"""
PhotoMind Dashboard API Server

Lightweight Flask backend that wraps the existing PhotoMind tools
and serves photo files (with HEIC→JPEG conversion) for the React frontend.
"""

import io
import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"])

KB_PATH = PROJECT_ROOT / "knowledge_base" / "photo_index.json"
PHOTOS_DIR = PROJECT_ROOT / "photos"
EVAL_PATH = PROJECT_ROOT / "eval" / "results" / "eval_results.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.route("/api/knowledge-base", methods=["GET"])
def get_knowledge_base():
    """Return the full photo knowledge base."""
    data = _load_json(KB_PATH)
    if not data:
        return jsonify({"error": "Knowledge base not found. Run ingestion first."}), 404
    return jsonify(data)


@app.route("/api/photos/<filename>", methods=["GET"])
def serve_photo(filename: str):
    """Serve a photo file. Converts HEIC to JPEG on the fly."""
    photo_path = PHOTOS_DIR / filename
    if not photo_path.exists():
        abort(404)

    # Security: ensure the resolved path is inside PHOTOS_DIR
    if not photo_path.resolve().is_relative_to(PHOTOS_DIR.resolve()):
        abort(403)

    suffix = photo_path.suffix.lower()
    if suffix in (".heic", ".heif"):
        try:
            import PIL.Image
            import pillow_heif

            pillow_heif.register_heif_opener()
            img = PIL.Image.open(photo_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
        except Exception as e:
            return jsonify({"error": f"Failed to convert HEIC: {str(e)}"}), 500
    else:
        return send_file(photo_path)


@app.route("/api/query", methods=["POST"])
def quick_query():
    """Run a quick query using PhotoKnowledgeBaseTool directly (no LLM cost)."""
    body = request.get_json(silent=True) or {}
    query_text = body.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

        tool = PhotoKnowledgeBaseTool()
        tool.knowledge_base_path = str(KB_PATH)
        result_json = tool._run(
            query=query_text,
            query_type=body.get("query_type", "auto"),
            top_k=body.get("top_k", 5),
            confidence_threshold=body.get("confidence_threshold", 0.15),
        )
        return jsonify(json.loads(result_json))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query/crew", methods=["POST"])
def crew_query():
    """Run a full CrewAI query (uses LLM — slower and costs money)."""
    body = request.get_json(silent=True) or {}
    query_text = body.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        from src.crews.query_crew import create_query_crew

        crew = create_query_crew()
        result = crew.kickoff(inputs={"user_query": query_text})
        return jsonify({"result": str(result), "query": query_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eval-results", methods=["GET"])
def get_eval_results():
    """Return evaluation results."""
    data = _load_json(EVAL_PATH)
    if not data:
        return jsonify({"error": "No evaluation results found. Run eval first."}), 404
    return jsonify(data)


if __name__ == "__main__":
    print(f"PhotoMind API starting...")
    print(f"  Knowledge base: {KB_PATH}")
    print(f"  Photos dir:     {PHOTOS_DIR}")
    print(f"  Eval results:   {EVAL_PATH}")
    app.run(debug=True, port=5001)
