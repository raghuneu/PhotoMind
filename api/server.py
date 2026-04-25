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
import re
import time
import base64
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env file so OPENAI_API_KEY is available for CrewAI

from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
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

_default_origins = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
_allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Auth ───────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_configured_api_key() -> str:
    """Load the API key from settings (cached after first call)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import get_settings
    return get_settings().api_key


async def verify_api_key(api_key: str | None = Security(_api_key_header)):
    """Dependency that enforces API key auth when configured."""
    expected = _get_configured_api_key()
    if not expected:  # No key configured — auth disabled
        return
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── In-memory caches ───────────────────────────────────────────────────
_kb_cache: dict | None = None
_repo = None  # PhotoRepository singleton for default user
_user_repos: dict[str, object] = {}  # user_id → PhotoRepository


class LRUQueryCache:
    """Simple TTL-aware LRU cache for fast query results."""

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 300):
        self._cache: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def get(self, key: tuple) -> dict | None:
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def put(self, key: tuple, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), value)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


_query_cache = LRUQueryCache(maxsize=128, ttl_seconds=300)


def _get_repo(user_id: str | None = None):
    """Return a repository instance, optionally scoped to a user.

    When ``user_id`` is provided, creates a per-user Qdrant collection
    (``photos_{user_id}``) and a per-user JSON KB file.  The default
    (no user_id) returns the global singleton.
    """
    if user_id is None:
        global _repo
        if _repo is None:
            try:
                import sys
                sys.path.insert(0, str(PROJECT_ROOT))
                from src.storage import get_repository
                _repo = get_repository()
            except Exception:
                _repo = None
        return _repo

    # Per-user scoped repo
    if user_id in _user_repos:
        return _user_repos[user_id]

    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.storage import get_repository
        collection = f"photos_{user_id}"
        kb_path = PROJECT_ROOT / "knowledge_base" / f"photo_index_{user_id}.json"
        repo = get_repository(collection=collection, kb_path=str(kb_path))
        _user_repos[user_id] = repo
        return repo
    except Exception:
        return None


def _load_kb() -> dict:
    global _kb_cache
    if _kb_cache is None:
        repo = _get_repo()
        if repo is not None:
            try:
                _kb_cache = {
                    "metadata": repo.metadata(),
                    "photos": repo.all_photos(),
                }
                return _kb_cache
            except Exception:
                pass
        # Fallback to direct JSON read
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
    confidence_score: float = 0.0


# ── Routes: Query ────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    _=Depends(verify_api_key),
    x_user_id: str | None = Header(None),
):
    """Execute a query against the PhotoMind knowledge base."""
    start = time.time()

    if req.mode == "fast":
        return _fast_query(req, start, user_id=x_user_id)
    elif req.mode == "full":
        return _full_query(req, start)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}. Use 'fast' or 'full'.")


def _fast_query(req: QueryRequest, start: float, user_id: str | None = None) -> dict:
    """Direct Python search — no OpenAI calls, free, <1s. Cached."""
    cache_key = (req.query, req.query_type, req.top_k, user_id)
    cached = _query_cache.get(cache_key)
    if cached is not None:
        cached["latency_s"] = round(time.time() - start, 3)
        return cached

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

    repo = _get_repo(user_id=user_id)
    tool = PhotoKnowledgeBaseTool(
        knowledge_base_path=str(KB_PATH),
        repository=repo,
    )
    raw = tool._run(
        query=req.query,
        query_type=req.query_type,
        top_k=req.top_k,
    )
    parsed = json.loads(raw)
    elapsed = time.time() - start

    result = {
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
    _query_cache.put(cache_key, result)
    return result


def _parse_crew_confidence(result_text: str) -> tuple[str, float]:
    """Extract confidence grade and score from CrewAI crew output text.

    Returns (grade, score) with sensible defaults if parsing fails.
    """
    text = result_text.lower()
    text_clean = re.sub(r'\*+', '', text)

    # Try JSON parse first
    try:
        json_text = result_text.strip()
        if json_text.startswith("```"):
            json_text = json_text.split("\n", 1)[1] if "\n" in json_text else json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3].strip()
        data = json.loads(json_text)
        if isinstance(data, dict):
            grade = data.get("confidence_grade", "").upper()
            if grade in ("A", "B", "C", "D", "F"):
                score = float(data.get("confidence_score", 0.0))
                return grade, score
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex extraction for grade
    grade = "B"  # default fallback
    for g in ["A", "B", "C", "D", "F"]:
        gl = g.lower()
        if (
            re.search(rf'\bconfidence[_ ]?grade[^:]*:\s*["\']?{gl}["\']?', text_clean)
            or re.search(rf'\bgrade[^a-z]*:\s*["\']?{gl}["\']?\b', text_clean)
            or f'"confidence_grade": "{gl}"' in text
            or f"confidence_grade: {gl}" in text_clean
        ):
            grade = g
            break

    # Regex extraction for score
    score = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "F": 0.1}.get(grade, 0.5)
    score_match = re.search(r'confidence_score["\s:]+([0-9]+\.[0-9]+)', text)
    if not score_match:
        score_match = re.search(r'confidence["\s:]+([0-9]+\.[0-9]+)', text)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            pass

    return grade, score


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
        grade, score = _parse_crew_confidence(result_text)
        photos = re.findall(r'[\w\-]+\.(?:jpg|jpeg|png|webp|heic)', result_text, re.IGNORECASE)
        source_photos = list(dict.fromkeys(p.lower() for p in photos))  # dedupe, preserve order

        # Resolve lowercased filenames → KB UUIDs so thumbnail/image endpoints work
        kb = _load_kb()
        fname_to_id = {p["filename"].lower(): p["id"] for p in kb.get("photos", [])}

        # Build result objects from extracted photo filenames so the UI can
        # render photo cards for Deep Reasoning results (previously empty).
        crew_results = []
        for i, photo_path in enumerate(source_photos):
            crew_results.append({
                "photo_id": fname_to_id.get(photo_path, photo_path),
                "photo_path": photo_path,
                "relevance_score": round(max(score - i * 0.05, 0.1), 3),
                "evidence": f"Identified by CrewAI deep reasoning pipeline",
                "image_type": "unknown",
            })

        return {
            "query": req.query,
            "mode": "full",
            "query_type_detected": "crewai_hierarchical",
            "results": crew_results,
            "confidence_grade": grade,
            "confidence_score": score,
            "answer_summary": result_text[:2000],
            "source_photos": source_photos,
            "warning": None,
            "latency_s": round(elapsed, 3),
            "routing_source": "crewai_hierarchical_process",
        }
    except Exception as e:
        raise HTTPException(500, f"CrewAI query failed: {str(e)}")


# ── Routes: Streaming Query (SSE) ────────────────────────────────────────

def _format_sse(event: str, data: dict) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_fast_query(req: "QueryRequest", user_id: str | None):
    """Stream a fast-mode query as SSE events.

    Emits: routing → retrieval → token* → done. The LLM is never invoked in
    fast mode; tokens here are chunks of the deterministic answer_summary
    so the UI can render progressively. Free; perceived latency <200 ms.
    """
    import asyncio
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

    start = time.time()

    # Cache hit: emit final event immediately
    cache_key = (req.query, req.query_type, req.top_k, user_id)
    cached = _query_cache.get(cache_key)
    if cached is not None:
        yield _format_sse("routing", {"source": "cache", "query_type_detected": cached.get("query_type_detected")})
        yield _format_sse("done", {**cached, "latency_s": round(time.time() - start, 3), "cached": True})
        return

    yield _format_sse("routing", {"stage": "classifying", "query": req.query})
    await asyncio.sleep(0)  # cooperative yield

    repo = _get_repo(user_id=user_id)
    tool = PhotoKnowledgeBaseTool(knowledge_base_path=str(KB_PATH), repository=repo)
    raw = tool._run(query=req.query, query_type=req.query_type, top_k=req.top_k)
    parsed = json.loads(raw)

    yield _format_sse("retrieval", {
        "query_type_detected": parsed.get("query_type_detected"),
        "result_count": len(parsed.get("results", [])),
        "confidence_grade": parsed.get("confidence_grade"),
    })

    summary = parsed.get("answer_summary", "") or ""
    # Chunk into ~40-char pieces for progressive rendering
    chunk_size = 40
    for i in range(0, len(summary), chunk_size):
        yield _format_sse("token", {"text": summary[i:i + chunk_size]})
        await asyncio.sleep(0.02)

    final = {
        "query": req.query,
        "mode": "fast",
        "query_type_detected": parsed.get("query_type_detected", "unknown"),
        "results": parsed.get("results", []),
        "confidence_grade": parsed.get("confidence_grade", "F"),
        "confidence_score": parsed.get("confidence_score", 0.0),
        "answer_summary": summary,
        "source_photos": parsed.get("source_photos", []),
        "warning": parsed.get("warning"),
        "latency_s": round(time.time() - start, 3),
        "routing_source": parsed.get("routing_rationale", ""),
    }
    _query_cache.put(cache_key, final)
    yield _format_sse("done", {**final, "cached": False})


async def _stream_full_query(req: "QueryRequest"):
    """Stream a full-mode (CrewAI + GPT-4o) query as SSE events.

    Wraps the blocking crew.kickoff() in a threadpool and emits stage events
    around it. Does NOT reduce wall-clock latency, but surfaces progress to
    the UI so users see activity instead of a 45-second spinner.
    """
    import asyncio
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))

    start = time.time()
    yield _format_sse("routing", {"stage": "crewai_bootstrapping"})
    try:
        from src.crews.query_crew import create_query_crew
        crew = create_query_crew()
    except Exception as e:
        yield _format_sse("error", {"message": f"Crew init failed: {e}"})
        return

    yield _format_sse("agent_step", {"agent": "Controller", "stage": "classifying"})

    def _blocking_kickoff():
        return crew.kickoff(inputs={"user_query": req.query})

    try:
        result = await asyncio.to_thread(_blocking_kickoff)
    except Exception as e:
        yield _format_sse("error", {"message": f"CrewAI query failed: {e}"})
        return

    yield _format_sse("agent_step", {"agent": "Synthesizer", "stage": "completed"})
    result_text = str(result)
    grade, score = _parse_crew_confidence(result_text)
    photos = re.findall(r'[\w\-]+\.(?:jpg|jpeg|png|webp|heic)', result_text, re.IGNORECASE)
    source_photos = list(dict.fromkeys(p.lower() for p in photos))

    # Resolve lowercased filenames → KB UUIDs so thumbnail/image endpoints work
    kb = _load_kb()
    fname_to_id = {p["filename"].lower(): p["id"] for p in kb.get("photos", [])}

    # Build result objects so the UI can render photo cards
    crew_results = []
    for i, photo_path in enumerate(source_photos):
        crew_results.append({
            "photo_id": fname_to_id.get(photo_path, photo_path),
            "photo_path": photo_path,
            "relevance_score": round(max(score - i * 0.05, 0.1), 3),
            "evidence": "Identified by CrewAI deep reasoning pipeline",
            "image_type": "unknown",
        })

    yield _format_sse("done", {
        "query": req.query,
        "mode": "full",
        "query_type_detected": "crewai_hierarchical",
        "answer_summary": result_text[:2000],
        "confidence_grade": grade,
        "confidence_score": score,
        "source_photos": source_photos,
        "results": crew_results,
        "warning": None,
        "latency_s": round(time.time() - start, 3),
        "routing_source": "crewai_hierarchical_process",
    })


@app.post("/api/query/stream")
async def query_stream(
    req: QueryRequest,
    _=Depends(verify_api_key),
    x_user_id: str | None = Header(None),
):
    """SSE streaming variant of /api/query.

    Fast mode: streams routing/retrieval/token events and a final `done` event
    (zero OpenAI cost). Full mode: streams coarse agent-step events around the
    blocking CrewAI call. Consume with an EventSource client or `curl -N`.
    """
    if req.mode == "fast":
        generator = _stream_fast_query(req, user_id=x_user_id)
    elif req.mode == "full":
        generator = _stream_full_query(req)
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}. Use 'fast' or 'full'.")
    return StreamingResponse(generator, media_type="text/event-stream")


# ── Routes: Knowledge Base ───────────────────────────────────────────────

@app.get("/api/knowledge-base")
async def get_knowledge_base(x_user_id: str | None = Header(None)):
    """Return the full knowledge base metadata and photo list."""
    if x_user_id:
        repo = _get_repo(user_id=x_user_id)
        if repo is not None:
            try:
                photos = repo.all_photos()
                return {
                    "metadata": repo.metadata(),
                    "total_photos": len(photos),
                    "photos": photos,
                }
            except Exception:
                pass
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
        return JSONResponse(
            content={"photo_id": photo_id, "data_url": f"data:image/jpeg;base64,{b64}"},
            headers={"Cache-Control": "public, max-age=86400"}  # 24 hour cache
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to generate thumbnail: {e}")


@app.get("/api/photos/{photo_id}/image")
async def get_photo_image(photo_id: str):
    """Return full-size image as base64 data URL with caching."""
    kb = _load_kb()
    if not kb:
        raise HTTPException(503, "Knowledge base not loaded")

    # Find photo entry (field is "id", not "photo_id")
    photo = next((p for p in kb.get("photos", []) if p["id"] == photo_id), None)
    if not photo:
        raise HTTPException(404, f"Photo not found: {photo_id}")

    file_path = PROJECT_ROOT / photo["file_path"]
    if not file_path.exists():
        raise HTTPException(404, f"Image file not found: {photo['file_path']}")

    try:
        from PIL import Image
        import io

        # Handle HEIC format
        suffix = file_path.suffix.lower()
        if suffix == ".heic":
            import pillow_heif
            pillow_heif.register_heif_opener()

        with Image.open(file_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Keep original size but optimize quality
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return JSONResponse(
                content={
                    "photo_id": photo_id,
                    "data_url": f"data:image/jpeg;base64,{b64}",
                    "width": img.width,
                    "height": img.height
                },
                headers={"Cache-Control": "public, max-age=604800"}  # 7 day cache
            )
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {e}")


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
async def submit_feedback(req: FeedbackRequest, _=Depends(verify_api_key)):
    """Record user feedback for the adaptive feedback loop."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.tools.feedback_store import FeedbackStore
    store = FeedbackStore(path=str(FEEDBACK_PATH))
    store.record_outcome(
        query=req.query,
        query_type=req.strategy,
        correct=req.was_correct,
        confidence_score=req.confidence_score,
    )
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
                "crew": "query",
                "description": "Receives the user query, classifies intent, and delegates to specialist agents (planning=True, hierarchical).",
            },
            {
                "name": "Photo Analyst",
                "role": "specialist",
                "crew": "ingestion",
                "description": "Analyzes photo content with GPT-4o Vision during ingestion — produces OCR, description, entities, and image_type.",
            },
            {
                "name": "Knowledge Retriever",
                "role": "specialist",
                "crew": "shared",
                "description": "Searches the knowledge base via PhotoKnowledgeBaseTool (4 strategies, RL-routed). Used in both ingestion and query crews.",
            },
            {
                "name": "Insight Synthesizer",
                "role": "specialist",
                "crew": "query",
                "description": "Combines retrieved evidence into a graded A–F answer with source attribution (query crew only).",
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
                "purpose": "Query routing — selects which search strategy (arm) to use",
                "state": (
                    "KMeans cluster id (k=4) over 396-dim hybrid query vector "
                    "(12 handcrafted features + 384-dim MiniLM embedding, sentence-transformers/all-MiniLM-L6-v2)"
                ),
                "actions": "4 arms (factual, semantic, behavioral, embedding)",
                "training": "Offline PhotoMindSimulator, 4000 episodes × 5 seeds, zero API cost",
            },
            {
                "name": "DQN Confidence Calibrator",
                "type": "Deep Q-Network (FC 8→64→64→5)",
                "purpose": "Confidence grading — decides whether to accept, hedge, requery, or decline",
                "state": "8-dim state (top score, result count, score spread, strategy idx, query features)",
                "actions": "5 actions (accept_high, accept_moderate, hedge, requery, decline)",
                "training": "Offline PhotoMindSimulator, 4000 episodes × 5 seeds, silent-failure penalty −1.0",
            },
        ],
        "pipeline_modes": [
            {"name": "fast", "description": "Direct Qdrant vector + keyword search with RL routing. No LLM calls. <1 s.", "cost": "Free"},
            {"name": "full", "description": "CrewAI hierarchical pipeline (Controller → Retriever → Synthesizer) with GPT-4o. ~40 s.", "cost": "~$0.01–0.05/query"},
        ],
        "storage": {
            "backend": "Qdrant (vector) with JSON fallback",
            "collection": "photos",
            "embedding_model": "all-MiniLM-L6-v2 (384-dim)",
            "fallback": "JsonPhotoRepository reads knowledge_base/photo_index.json",
        },
    }


@app.post("/api/cache/clear")
async def clear_cache(_=Depends(verify_api_key)):
    """Clear the fast-query LRU cache."""
    _query_cache.clear()
    return {"status": "ok", "message": "Query cache cleared"}


@app.get("/api/health")
async def health():
    """Health check."""
    kb = _load_kb()
    repo = _get_repo()
    qdrant_status = "disconnected"
    if repo is not None:
        try:
            qdrant_status = f"ok ({repo.__class__.__name__}, {repo.photo_count()} photos)"
        except Exception:
            qdrant_status = "error"
    return {
        "status": "ok",
        "knowledge_base_photos": len(kb.get("photos", [])),
        "has_eval_results": EVAL_RESULTS_PATH.exists(),
        "has_ablation_results": ABLATION_PATH.exists(),
        "has_rl_models": (PROJECT_ROOT / "knowledge_base" / "rl_models" / "bandit_thompson.pkl").exists(),
        "storage_backend": qdrant_status,
        "query_cache_size": len(_query_cache._cache),
    }
