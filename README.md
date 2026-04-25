# PhotoMind

> **Take-Home Final: Reinforcement Learning for Agentic AI Systems**
> This project is the submission for the take-home final. The work lives on the `feature/reinforcement-learning-extension` branch

A multimodal personal photo knowledge retrieval system built with CrewAI. Turns your phone's photo library into a queryable knowledge base — ask natural-language questions, get answers with confidence scores and source attribution.

**Repository:** [github.com/raghuneu/PhotoMind](https://github.com/raghuneu/PhotoMind/tree/feature/reinforcement-learning-extension) · **Demo:** [youtu.be/UQRdkW2mAgc](https://www.youtube.com/watch?v=UQRdkW2mAgc)

[![Demo Video](Photomind_withRL.png)](https://www.youtube.com/watch?v=UQRdkW2mAgc)

## What It Does

PhotoMind uses GPT-4o Vision to analyze personal photos (bills, receipts, food photos, screenshots, documents) and builds a searchable knowledge base backed by Qdrant vector DB with hybrid search (dense embedding ANN + keyword matching via Reciprocal Rank Fusion). Three query modes:

- **Factual** — "How much did I spend at ALDI?" → extracts $18.69 from receipt OCR
- **Semantic** — "Show me photos of pizza" → matches against visual descriptions
- **Behavioral** — "What type of food do I photograph most?" → aggregates patterns across all photos

Every answer includes a confidence grade (A–F) and cites the specific source photo.

## Architecture

```
INGESTION CREW (Process.sequential)
  [Scan photos/] → [Analyze with GPT-4o Vision] → [Build JSON knowledge base]
  (--direct flag available for faster batch processing via direct API calls)
  Dual-write: ingests into both JSON file and Qdrant vector DB simultaneously

QUERY CREW (Process.hierarchical, manager-delegated)
  [Controller] classifies query intent → delegates to specialists
    ├── Task 1: Knowledge Retriever — searches KB with PhotoKnowledgeBaseTool
    └── Task 2: Insight Synthesizer — synthesizes answer with confidence grade + citation

STORAGE LAYER (Repository Pattern)
  ├── QdrantPhotoRepository (default) — vector DB with hybrid search (RRF)
  └── JsonPhotoRepository (fallback)  — flat JSON file for offline/RL training

API SERVER (FastAPI)
  uvicorn api.server:app — REST API with LRU cache, API key auth, multi-user scoping

FEEDBACK LOOP (persistent, adaptive)
  [Eval results] → [FeedbackStore] → adjusts confidence thresholds per strategy
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| Controller | Orchestrates query routing; classifies intent (factual/semantic/behavioral) | — (manager) |
| Photo Analyst | Analyzes images: OCR, entity extraction, classification | PhotoVisionTool, DirectoryReadTool |
| Knowledge Retriever | Searches knowledge base; retrieves evidence | PhotoKnowledgeBaseTool (custom), FileReadTool |
| Insight Synthesizer | Synthesizes grounded answers with confidence grading | FileReadTool, SerperDevTool (optional) |

### Tools

| Tool | Type | Purpose |
|------|------|---------|
| `PhotoVisionTool` | Custom | GPT-4o Vision wrapper with HEIC support (`src/tools/photo_vision.py`) |
| `PhotoKnowledgeBaseTool` | Custom | 3-strategy search with feedback integration (`src/tools/photo_knowledge_base.py`) |
| `DirectoryReadTool` | Built-in | Scans the photos directory |
| `FileReadTool` | Built-in | Reads the knowledge base file |
| `JSONSearchTool` | Built-in | Embedding-based JSON search (sentence-transformers) |
| `SerperDevTool` | Built-in (optional) | Web search enrichment |

## Setup

### Requirements

- Python 3.10–3.13 (CrewAI requires `< 3.14`)
- OpenAI API key (GPT-4o access)

### 1. Create a virtual environment

```bash
# If your system Python is 3.14+, use pyenv:
~/.pyenv/versions/3.10.14/bin/python3 -m venv .venv
source .venv/bin/activate

# Otherwise:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

First run downloads the `all-MiniLM-L6-v2` sentence-transformer model (~80MB) for local embeddings.

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

**Required:**
- `OPENAI_API_KEY` — GPT-4o Vision + agent reasoning

**Optional:**
- `SERPER_API_KEY` — enables web search enrichment in answers
- `REPOSITORY_BACKEND` — `json` (default) or `qdrant` for vector DB
- `QDRANT_URL` — Qdrant server address (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` — Qdrant collection name (default: `photos`)
- `API_KEY` — when set, protects POST endpoints with `X-API-Key` header auth

### 4. Create directories and add photos

```bash
mkdir -p photos knowledge_base eval/results
```

Place 15–25 photos in `photos/`. iPhone photos (HEIC format) are fully supported. Recommended mix for best demo coverage:
- Grocery/restaurant receipts (3–5)
- Food photos (3–5)
- Screenshots from apps or websites (1–2)
- Documents or notes (1–2)

### 5. Start Qdrant (vector database)

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Qdrant runs as a local Docker container. The ingestion pipeline auto-creates the `photos` collection on first write. If Qdrant is unavailable, the system falls back to the JSON file backend automatically.

## Usage

### Ingest photos

```bash
# Default: CrewAI multi-agent pipeline
python -m src.main ingest

# Fast mode: direct API calls, bypasses CrewAI agents
python -m src.main ingest --direct
```

Analyzes all photos in `photos/` using GPT-4o Vision and writes to both `knowledge_base/photo_index.json` and the Qdrant vector DB (dual-write). Idempotent — re-running skips already-indexed photos. The default mode uses CrewAI agents for orchestrated ingestion; `--direct` is faster for batch processing.

### Query the knowledge base

```bash
# Factual — extract specific facts
python -m src.main query "How much did I spend at ALDI?"
python -m src.main query "What is the address on my Trader Joe's receipt?"

# Semantic — find by visual description
python -m src.main query "Show me photos of pizza"
python -m src.main query "Find photos that feel like summer"

# Behavioral — analyze patterns
python -m src.main query "What type of food do I photograph most?"
python -m src.main query "Which store do I shop at most often?"

# Edge cases — system should decline gracefully
python -m src.main query "What was my electric bill?"  # not in library

# Fast path — skip CrewAI, run retrieval + routing directly
# Zero OpenAI cost, sub-second latency, great for scripts and CI.
python -m src.main query --direct "How much did I spend at ALDI?"
```

### Run the API server

```bash
uvicorn api.server:app --reload --port 8000
```

The FastAPI server exposes the knowledge base and RL models as a REST API. Two query modes:

- **Fast** (`mode: "fast"`) — direct Python search with RL routing, no OpenAI calls, <1s, free
- **Full** (`mode: "full"`) — CrewAI pipeline with GPT-4o reasoning, 15–45s, ~$0.01–0.05/query

Key endpoints:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/query` | POST | API key | Query the knowledge base |
| `/api/query/stream` | POST | API key | SSE streaming variant — emits `routing`/`retrieval`/`token`/`done` events for progressive UI rendering |
| `/api/knowledge-base` | GET | — | List all indexed photos |
| `/api/health` | GET | — | Health check with backend status |
| `/api/cache/clear` | POST | API key | Clear the LRU query cache |
| `/api/feedback` | POST | API key | Submit feedback for adaptive thresholds |
| `/api/eval/results` | GET | — | Latest evaluation results |

**Headers:**
- `X-API-Key` — required for POST endpoints when `API_KEY` is set in `.env`
- `X-User-Id` — optional; scopes queries to a per-user Qdrant collection and JSON KB

Example query:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did I spend at ALDI?", "mode": "fast"}'
```

### Run the evaluation suite

```bash
python -m src.main eval
```

Runs 20 test queries across 4 categories and reports:
- Retrieval Accuracy — was the correct source photo found?
- Routing Accuracy — was the query intent correctly detected?
- Silent Failure Rate — did the system ever confidently return a wrong answer?
- Decline Accuracy — were impossible queries correctly refused?

Results saved to `eval/results/eval_results.json`. Run history appended to `eval/results/eval_history.json` for trend analysis. Each run also feeds back into the adaptive confidence threshold system via `FeedbackStore`.

### Train the RL components (offline, no API calls)

```bash
# Train both components across 5 seeds (4000 episodes each, ~120s total)
python -m src.main train

# Train with custom episode count
python -m src.main train 1000

# Run full RL evaluation: 5 configs x 5 seeds x 56 queries
python -m src.main rl-eval

# Run 7-config ablation study → eval/results/ablation_results.json
python -m src.main ablation
```

RL training requires no API keys — it runs entirely on the cached knowledge base (zero API cost).
Requires an existing `knowledge_base/photo_index.json` — run `python -m src.main ingest` first.
Trained models are saved to `knowledge_base/rl_models/` and loaded automatically at query time.

## Reinforcement Learning Extension

PhotoMind integrates two RL approaches that replace rule-based components:

### Approach 1: Contextual Bandits — Query Routing (Exploration Strategies)

Replaces the keyword-based `_classify_query()` in `PhotoKnowledgeBaseTool` with a learned policy that selects the optimal search strategy (factual / semantic / behavioral) based on query features.

- **ThompsonSamplingBandit** — Beta posterior per context cluster, provably optimal exploration
- **UCBBandit** — UCB1 upper confidence bound per cluster
- **EpsilonGreedyBandit** — Baseline comparison
- Context clustering via KMeans on 396-dimensional hybrid query feature vectors (12 handcrafted + 384 MiniLM embedding dims)
- Training: 4000 episodes × 5 seeds on offline cached search results (zero API cost)

### Approach 2: DQN — Confidence Calibration (Value-Based Learning)

Replaces static confidence thresholding with a DQN that learns when to accept, hedge, or decline retrieval results — directly addressing the silent failure problem.

- Architecture: FC(8→64) → ReLU → FC(64→64) → ReLU → FC(64→5), adapted from the LunarLander DQN (extended with a requery action)
- State: 8-dim vector (top score, score gap, result count, strategy index, query features, entity match indicators)
- Actions: `accept_high`, `accept_moderate`, `hedge`, `requery`, `decline`
- Reward: penalty matrix that heavily penalizes silent failures (confident-but-wrong answers)

### RL Results (56 test queries, 5 seeds)

| Config | Retrieval | Routing | Silent Failures | Decline Acc |
|--------|-----------|---------|-----------------|-------------|
| Baseline (Rule-Based) | 87.5% | 76.8% | 1.8% | 90.9% |
| Full RL (Thompson+DQN) | 87.5% | 67.1% | **0.0%** | **98.2%** |

Key finding: The DQN eliminates silent failures (Full RL: 0.0% vs 1.8% baseline; p < 0.0001). The bandit trades routing accuracy for silent failure reduction on ambiguous queries where the "correct" routing label is itself ambiguous.

*Note: Numbers above are from the 5-config rl-eval harness. The full 7-config ablation in the technical report uses a separate run and shows slightly different values (e.g., 96.4% decline, 1.1% DQN-Only silent failure) due to the strategy_type_map correction applied before the ablation run.*

### RL Architecture

```
User Query
    │
    ▼
[QueryFeatureExtractor]  →  396-dim hybrid feature vector
    │
    ▼
[ContextualBandit]  →  selects arm: factual | semantic | behavioral
    │                   (ThompsonSampling / UCB / epsilon-greedy)
    ▼
[PhotoKnowledgeBaseTool]  →  runs selected strategy, returns results
    │
    ▼
[ConfidenceState]  →  8-dim state vector from retrieval results
    │
    ▼
[ConfidenceDQN]  →  action: accept_high | accept_moderate | hedge | requery | decline
    │
    ▼
[Insight Synthesizer]  →  graded answer with source attribution
```

**Offline simulation training:** Both components are trained using `PhotoMindSimulator`, which pre-computes all 3 search strategies on all 56 queries once (zero API calls). Training 4000 episodes × 5 seeds × 2 components takes ~120 seconds on CPU.

## Custom Tool: PhotoKnowledgeBaseTool

`src/tools/photo_knowledge_base.py` — the core differentiator.

**Three search strategies selected automatically by query intent:**

| Strategy | Trigger keywords | How it works |
|----------|-----------------|-------------|
| Factual | "how much", "date", "address", "items", "vendor" | Entity matching + OCR text search |
| Semantic | Default (no other match) | Keyword overlap on descriptions, normalized by meaningful words |
| Behavioral | "most", "often", "how many", "breakdown", "pattern" | Frequency aggregation across all photos |

**Output always includes:** confidence grade (A–F), numeric score (0–1), source photo filenames, and a plain-language summary.

**Input schema (Pydantic):**
```python
query: str               # Natural language question
query_type: str = "auto" # Force strategy or let tool classify
top_k: int = 3           # Number of results
confidence_threshold: float = 0.15  # Minimum score to include
```

## Evaluation Results

### Base System (25 photos, 20 queries — with Qdrant hybrid search)

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | **100%** |
| Routing Accuracy | **85%** |
| Silent Failure Rate | **0%** |
| Decline Accuracy | **100%** |
| Avg Latency | ~46s/query |

### RL Extension (25 photos, 56 queries, 5 seeds)

| Config | Retrieval | Routing | Silent Fail | Decline Acc |
|--------|-----------|---------|-------------|-------------|
| Baseline (rule-based) | 87.5% | 76.8% | 1.8% | 90.9% |
| Bandit Only (Thompson) | 86.8% | 66.4% | **0.0%** | 81.8% |
| DQN Only | 87.5% | 76.8% | **0.4%** | **100.0%** |
| Full RL (Thompson+DQN) | 87.5% | 67.1% | **0.0%** | **98.2%** |

Statistical tests (Full RL vs Baseline): silent failure p < 0.0001 (***), decline accuracy p = 0.016 (*)

## Project Structure

```
PhotoMind/
├── api/
│   └── server.py                    # FastAPI REST API (LRU cache, auth, multi-user)
├── src/
│   ├── main.py                      # CLI entry point (ingest / query / eval / train / rl-eval / ablation)
│   ├── config.py                    # Pydantic settings (reads .env)
│   ├── ingest_direct.py             # Direct ingestion (1 API call/photo, dual-write)
│   ├── agents/
│   │   └── definitions.py           # 4 agent factory functions
│   ├── tasks/
│   │   ├── ingestion.py             # Scan, analyze, index tasks
│   │   └── query.py                 # Query task with intent routing
│   ├── crews/
│   │   ├── ingestion_crew.py        # Sequential ingestion pipeline
│   │   └── query_crew.py            # Hierarchical query pipeline
│   ├── tools/
│   │   ├── photo_vision.py          # PhotoVisionTool (GPT-4o Vision + HEIC)
│   │   ├── photo_knowledge_base.py  # PhotoKnowledgeBaseTool (custom) — RL-enhanced
│   │   ├── query_memory.py          # Query memory and deduplication
│   │   └── feedback_store.py        # FeedbackStore (adaptive threshold learning)
│   ├── storage/
│   │   ├── __init__.py              # Public exports (PhotoRepository, get_repository)
│   │   ├── repository.py            # ABC + JsonPhotoRepository + QdrantPhotoRepository + factory
│   │   └── qdrant_client.py         # Qdrant connection helper (hybrid search, RRF)
│   └── rl/
│       ├── rl_config.py             # Centralized RL hyperparameters and reward matrix
│       ├── feature_extractor.py     # Query → 396-dim hybrid feature vector
│       ├── contextual_bandit.py     # Thompson Sampling, UCB, epsilon-greedy bandits
│       ├── dqn_confidence.py        # ConfidenceDQN and ConfidenceDQNAgent
│       ├── replay_buffer.py         # Experience replay buffer (adapted from LunarLander)
│       ├── reward.py                # Reward computation for bandit and DQN
│       ├── simulation_env.py        # Offline training environment (zero API cost)
│       └── training_pipeline.py     # Orchestrates training across seeds
├── eval/
│   ├── test_cases.py                # 20 original hand-labeled test queries
│   ├── expanded_test_cases.py       # 36 new cases (incl. 11 ambiguous) — 56 total
│   ├── novel_test_cases.py          # 15 intent-shift queries for robustness testing
│   ├── run_evaluation.py            # Base system evaluation harness
│   ├── run_rl_evaluation.py         # RL 5-config comparison harness
│   ├── ablation.py                  # 7-config ablation with paired t-tests
│   ├── statistical_analysis.py      # CI, paired t-test, Cohen's d utilities
│   └── results/                     # JSON results + eval history
├── viz/
│   ├── plot_learning_curves.py      # Bandit regret, DQN rewards, posteriors, epsilon decay
│   ├── plot_ablation.py             # Grouped bar chart (7 configs x 4 metrics)
│   ├── plot_regret.py               # Cumulative regret comparison (3 bandit types)
│   ├── plot_before_after.py         # Before/after RL comparison plots
│   ├── generate_diagrams.py         # Architecture and flow diagrams
│   └── figures/                     # Generated PNG and PDF figures
├── scripts/
│   ├── train_full.py                # Full training + optional ablation
│   ├── train_bandit.py              # Standalone bandit training
│   ├── train_dqn.py                 # Standalone DQN training
│   ├── precompute_cache.py          # Pre-compute search strategy cache
│   ├── scaling_benchmark.py         # Scaling and performance benchmarks
│   └── demo_comparison.py           # Rule-based vs RL before/after demo
├── knowledge_base/
│   ├── photo_index.json             # 25 indexed photos
│   └── rl_models/                   # Trained RL models
│       ├── bandit_thompson.pkl      # Trained Thompson Sampling bandit
│       └── dqn_confidence.pth       # Trained DQN confidence calibrator
├── tests/
│   ├── test_core.py                 # Core RL functionality tests (59 tests)
│   ├── test_search_strategies.py    # Search strategy correctness tests (24 tests)
│   └── test_repository.py           # Repository abstraction tests (13 tests)
├── web/                                 # React + TypeScript + Vite frontend (MUI, Recharts)
│   ├── src/
│   │   ├── App.tsx                      # Main app component
│   │   ├── components/                  # UI components
│   │   └── theme.ts                     # MUI theme configuration
│   ├── index.html
│   └── package.json
├── docs/
│   ├── figures/                         # Copied figures for GitHub Pages
│   ├── math_formulations.md             # Mathematical formulations for RL components
│   └── mermaid_diagrams/                # Mermaid diagram sources
├── Dockerfile                           # Multi-stage build (Node frontend + Python backend)
├── .dockerignore
├── .env.example
├── .gitignore
├── LICENSE
├── PROJECT_RETROSPECTIVE.md             # Project retrospective and lessons learned
├── requirements.txt
└── TECHNICAL_REPORT.md                  # Full technical documentation (base system + RL extension)
```

## Known Limitations

- Semantic search uses keyword overlap, not true vector embeddings — misses synonyms
- Qdrant backend requires Docker; falls back to flat JSON if unavailable
- Confidence grading is calibrated for a small corpus — thresholds may need tuning at scale
- RL bandit trained on 56 queries with 10x augmentation — may not generalize to unseen phrasing patterns outside the training distribution
- DQN requery action selects an alternate strategy randomly rather than learning which alternate to try; a learned requery policy could improve multi-step episode returns
- Bandit context clustering uses k=4 clusters on a small feature space — more data would support finer-grained contextualization
- Multi-user scoping creates separate Qdrant collections per user — no shared cross-user search
