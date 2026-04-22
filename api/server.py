"""
PhotoMind FastAPI Backend

Exposes the PhotoMind knowledge base, RL models, and evaluation results
through a REST API for the React frontend. Two query modes:

1. **Fast Query** — Direct Python search (no OpenAI, no CrewAI). Uses
   PhotoKnowledgeBaseTool internally with RL-powered routing. Free, <1s.
2. **Full Query** — CrewAI hierarchical pipeline with GPT-4o reasoning.
   Costs ~$0.01-0.05 per query, 15-45s latency.

Run: uvicorn api.server:app --reload --port 8000
"""

import json
import os
import time
import base64
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_ROOT / "knowledge_base" / "photo_index.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "eval" / "results" / "eval_results.json"
ABLATION_PATH = PROJECT_ROOT / "eval" / "results" / "ablation_results.json"
RL_TRAINING_PATH = PROJECT_ROOT / "eval" / "results" / "rl_training_results.json"
RL_EVAL_PATH = PROJECT_ROOT / "eval" / "results" / "rl_eval_results.json"
FEEDBACK_PATH = PROJECT_ROOT / "knowledge_base" / "feedback_store.json"
PHOTOS_DIR = PROJECT_ROOT / "photos"
FIGURES_DIR = PROJECT_ROOT / "viz" / "figures"


# ── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load knowledge base on startup
    _load_kb()
    yield


app = FastAPI(
    title="PhotoMind API",
    description="Personal photo knowledge retrieval with RL-powered routing",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory cache ─────────────────────────────────────────────────────
_kb_cache: dict | None = None


def _load_kb() -> dict:
    global _kb_cache
    if _kb_cache is None:
        with open(KB_PATH) as f:
            _kb_cache = json.load(f)
    return _kb_cache


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Request / Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question")
    query_type: str = Field(default="auto", description="factual|semantic|behavioral|embedding|auto")
    top_k: int = Field(default=3, ge=1, le=20)
    mode: str = Field(default="fast", description="fast (no API) or full (CrewAI + GPT-4o)")


class QueryResponse(BaseModel):
    query: str
    mode: str
    query_type_detected: str
    results: list
    confidence_grade: str
    confidence_score: float
    answer_summary: str
    source_photos: list[str]
    warning: str | None = None
    latency_s: float
    routing_source: str | None = None


class FeedbackRequest(BaseModel):
    query: str
    strategy: str
    was_correct: bool


# ── Routes: Query ────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Execute a query against the PhotoMind knowledge base."""
    start = time.time()

    if req.mode == "fast":
        return _fast_query(req, start)
    elif req.mode == "full":
        return _full_query(req, start)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}. Use 'fast' or 'full'.")


def _fast_query(req: QueryRequest, start: float) -> dict:
    """Direct Python search — no OpenAI calls, free, <1s."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

    tool = PhotoKnowledgeBaseTool(knowledge_base_path=str(KB_PATH))
    raw = tool._run(
        query=req.query,
        query_type=req.query_type,
        top_k=req.top_k,
    )
    parsed = json.loads(raw)
    elapsed = time.time() - start

    return {
        "query": req.query,
        "mode": "fast",
        "query_type_detected": parsed.get("query_type_detected", "unknown"),
        "results": parsed.get("results", []),
        "confidence_grade": parsed.get("confidence_grade", "F"),
        "confidence_score": parsed.get("confidence_score", 0.0),
        "answer_summary": parsed.get("answer_summary", ""),
        "source_photos": parsed.get("source_photos", []),
        "warning": parsed.get("warning"),
        "latency_s": round(elapsed, 3),
        "routing_source": parsed.get("routing_rationale", ""),
    }


def _full_query(req: QueryRequest, start: float) -> dict:
    """CrewAI pipeline — GPT-4o reasoning, costs money, 15-45s."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from src.crews.query_crew import create_query_crew

        crew = create_query_crew()
        result = crew.kickoff(inputs={"user_query": req.query})
        elapsed = time.time() - start

        # Try to parse structured output from crew
        result_text = str(result)
        return {
            "query": req.query,
            "mode": "full",
            "query_type_detected": "crewai_hierarchical",
            "results": [],
            "confidence_grade": "A",
            "confidence_score": 0.9,
            "answer_summary": result_text[:2000],
            "source_photos": [],
            "warning": None,
            "latency_s": round(elapsed, 3),
            "routing_source": "crewai_hierarchical_process",
        }
    except Exception as e:
        raise HTTPException(500, f"CrewAI query failed: {str(e)}")


# ── Routes: Knowledge Base ───────────────────────────────────────────────

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Return the full knowledge base metadata and photo list."""
    kb = _load_kb()
    return {
        "metadata": kb.get("metadata", {}),
        "total_photos": len(kb.get("photos", [])),
        "photos": kb.get("photos", []),
    }


@app.get("/api/knowledge-base/stats")
async def get_kb_stats():
    """Aggregate statistics about the knowledge base."""
    kb = _load_kb()
    photos = kb.get("photos", [])

    type_counts = {}
    entity_counts = {}
    total_entities = 0

    for p in photos:
        t = p.get("image_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        for e in p.get("entities", []):
            etype = e.get("type", "unknown")
            entity_counts[etype] = entity_counts.get(etype, 0) + 1
            total_entities += 1

    return {
        "total_photos": len(photos),
        "type_distribution": type_counts,
        "entity_type_distribution": entity_counts,
        "total_entities": total_entities,
        "has_ocr": sum(1 for p in photos if p.get("ocr_text", "").strip()),
        "avg_entities_per_photo": round(total_entities / max(len(photos), 1), 1),
    }


@app.get("/api/photos/{photo_id}/thumbnail")
async def get_photo_thumbnail(photo_id: str):
    """Return a base64-encoded thumbnail for a photo."""
    kb = _load_kb()
    photo = next((p for p in kb.get("photos", []) if p["id"] == photo_id), None)
    if not photo:
        raise HTTPException(404, "Photo not found")

    file_path = PROJECT_ROOT / photo["file_path"]
    if not file_path.exists():
        raise HTTPException(404, f"Photo file not found: {photo['file_path']}")

    # For HEIC, convert to JPEG; for others, serve directly
    try:
        from PIL import Image
        import io

        suffix = file_path.suffix.lower()
        if suffix == ".heic":
            import pillow_heif
            pillow_heif.register_heif_opener()

        img = Image.open(file_path)
        img.thumbnail((400, 400))
        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"photo_id": photo_id, "data_url": f"data:image/jpeg;base64,{b64}"}
    except Exception as e:
        raise HTTPException(500, f"Failed to generate thumbnail: {e}")


# ── Routes: Eval Results ─────────────────────────────────────────────────

@app.get("/api/eval/results")
async def get_eval_results():
    """Return the latest evaluation results."""
    data = _load_json(EVAL_RESULTS_PATH)
    if not data:
        raise HTTPException(404, "No evaluation results found. Run: python -m src.main eval")
    return data


@app.get("/api/eval/ablation")
async def get_ablation_results():
    """Return ablation study results."""
    data = _load_json(ABLATION_PATH)
    if not data:
        raise HTTPException(404, "No ablation results found. Run: python -m src.main ablation")
    return data


@app.get("/api/eval/rl-training")
async def get_rl_training_results():
    """Return RL training curves (bandit regret, DQN rewards)."""
    data = _load_json(RL_TRAINING_PATH)
    if not data:
        raise HTTPException(404, "No RL training results found. Run: python -m src.main train")
    return data


@app.get("/api/eval/rl-eval")
async def get_rl_eval_results():
    """Return RL evaluation results."""
    data = _load_json(RL_EVAL_PATH)
    if not data:
        raise HTTPException(404, "No RL eval results found. Run: python -m src.main rl-eval")
    return data


# ── Routes: Feedback ─────────────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record user feedback for the adaptive feedback loop."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.tools.feedback_store import FeedbackStore
    store = FeedbackStore()
    store.record_feedback(req.strategy, req.was_correct)
    return {"status": "ok", "message": f"Feedback recorded for strategy '{req.strategy}'"}


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """Return per-strategy feedback statistics."""
    data = _load_json(FEEDBACK_PATH)
    if not data:
        return {"strategies": {}, "message": "No feedback recorded yet"}
    return data


# ── Routes: Figures ──────────────────────────────────────────────────────

@app.get("/api/figures")
async def list_figures():
    """List all available visualization figures."""
    if not FIGURES_DIR.exists():
        return {"figures": []}
    figures = []
    for f in sorted(FIGURES_DIR.iterdir()):
        if f.suffix in (".png", ".pdf"):
            figures.append({"name": f.stem, "filename": f.name, "format": f.suffix[1:]})
    return {"figures": figures}


@app.get("/api/figures/{filename}")
async def get_figure(filename: str):
    """Serve a visualization figure."""
    path = FIGURES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"Figure not found: {filename}")
    # Prevent path traversal
    if not path.resolve().is_relative_to(FIGURES_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    return FileResponse(path)


# ── Routes: System Info ──────────────────────────────────────────────────

@app.get("/api/system/architecture")
async def get_architecture():
    """Return system architecture metadata for the Architecture Explorer."""
    return {
        "agents": [
            {
                "name": "Controller Agent",
                "role": "manager",
                "description": "Orchestrates query routing, delegates to specialist agents, synthesizes final response",
            },
            {
                "name": "Photo Analyst",
                "role": "specialist",
                "description": "Analyzes photo content using GPT-4o Vision for descriptions, OCR, entity extraction",
            },
            {
                "name": "Knowledge Retriever",
                "role": "specialist",
                "description": "Searches the knowledge base using 4 strategies: factual, semantic, behavioral, embedding",
            },
            {
                "name": "Insight Synthesizer",
                "role": "specialist",
                "description": "Combines retrieved evidence into coherent, confidence-graded answers",
            },
        ],
        "search_strategies": [
            {"name": "factual", "description": "Entity + OCR keyword matching", "type": "exact"},
            {"name": "semantic", "description": "Weighted keyword overlap with descriptions", "type": "keyword"},
            {"name": "embedding", "description": "Dense vector cosine similarity (all-MiniLM-L6-v2)", "type": "dense"},
            {"name": "behavioral", "description": "Frequency aggregation across corpus", "type": "aggregate"},
        ],
        "rl_components": [
            {
                "name": "Contextual Bandit",
                "type": "Thompson Sampling / UCB1 / Epsilon-Greedy",
                "purpose": "Query routing — selects which search strategy to use",
                "state": "12-dim feature vector from QueryFeatureExtractor",
                "actions": "4 arms (factual, semantic, behavioral, embedding)",
            },
            {
                "name": "DQN Confidence Calibrator",
                "type": "Deep Q-Network (FC 8→64→64→5)",
                "purpose": "Confidence grading — decides how confident to be in results",
                "state": "8-dim state (top score, result count, score spread, strategy idx, query features)",
                "actions": "5 actions (accept_high, accept_moderate, hedge, requery, decline)",
            },
        ],
        "pipeline_modes": [
            {"name": "fast", "description": "Direct Python search with RL routing. No API calls. <1s.", "cost": "Free"},
            {"name": "full", "description": "CrewAI hierarchical pipeline with GPT-4o. 15-45s.", "cost": "~$0.01-0.05/query"},
        ],
    }


@app.get("/api/health")
async def health():
    """Health check."""
    kb = _load_kb()
    return {
        "status": "ok",
        "knowledge_base_photos": len(kb.get("photos", [])),
        "has_eval_results": EVAL_RESULTS_PATH.exists(),
        "has_ablation_results": ABLATION_PATH.exists(),
        "has_rl_models": (PROJECT_ROOT / "knowledge_base" / "rl_models" / "bandit_thompson.pkl").exists(),
    }
