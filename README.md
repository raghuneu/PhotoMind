---
title: PhotoMind
emoji: 📸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

<div align="center">

# PhotoMind

**Multimodal Personal Photo Knowledge Retrieval System**

### [photomind.vercel.app](https://photomind.vercel.app/)

[![Python](https://img.shields.io/badge/Python-3.10–3.13-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agents-FF6F00?style=flat-square)](#)
[![GPT-4o](https://img.shields.io/badge/GPT--4o-Vision-412991?style=flat-square&logo=openai&logoColor=white)](#)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react&logoColor=black)](#)
[![RL](https://img.shields.io/badge/RL-DQN_+_Bandits-8B5CF6?style=flat-square)](#)

_Built for INFO 7375 — Prompt Engineering for Generative AI (Northeastern University, Spring 2026)_

**Raghu Ram Shanta Rajamani** — [shantharajamani.r@northeastern.edu](mailto:shantharajamani.r@northeastern.edu)

[What It Does](#what-it-does) · [The Stack](#the-stack) · [Architecture](#architecture) · [Setup](#setup) · [Usage](#usage) · [RL Extension](#reinforcement-learning-extension) · [Evaluation Results](#evaluation-results) · [Project Structure](#project-structure)

</div>

> This repository contains code for three incrementally developed assignments, each on its own branch:
>
> | Assignment                                                     | Branch                                                                                                                                  |
> | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
> | **Generative AI Project Assignment** (Final Project)           | [`main`](https://github.com/raghuneu/PhotoMind) ← _you are here_                                                                        |
> | Take-Home Final: Reinforcement Learning for Agentic AI Systems | [`feature/reinforcement-learning-extension`](https://github.com/raghuneu/PhotoMind/tree/feature/reinforcement-learning-extension)       |
> | Building Agentic Systems Assignment                            | [`feature/building_agentic_systems_assignment`](https://github.com/raghuneu/PhotoMind/tree/feature/building_agentic_systems_assignment) |

You took a photo of a grocery receipt last Tuesday. Now you need to know exactly what you spent. You could scroll through 500 photos, or you could just ask.

PhotoMind turns your phone's photo library into a queryable knowledge base. It analyzes photos with GPT-4o Vision, indexes them into a hybrid vector search engine, and answers natural-language questions with confidence scores and source attribution. Two RL components — contextual bandits for query routing and a DQN for confidence calibration — target silent failures (confident-but-wrong answers).

<p align="center">
  <a href="https://youtu.be/XMeceeWHjCE">
    <img src="Photomind.png" alt="PhotoMind Demo" width="90%"/>
  </a>
</p>

<p align="center">
  <strong>Repository:</strong> <a href="https://github.com/raghuneu/PhotoMind">github.com/raghuneu/PhotoMind</a> · <strong>Demo:</strong> <a href="https://youtu.be/XMeceeWHjCE">youtu.be/XMeceeWHjCE</a>
</p>

---

## What It Does

| Step              | What Happens                                                   | Details                                                                                     |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **1. Ingest**     | Scans `photos/` and analyzes each image with GPT-4o Vision     | OCR, entity extraction, classification. Dual-writes to JSON + Qdrant vector DB. Idempotent. |
| **2. Index**      | Builds a hybrid-searchable knowledge base                      | Dense embedding ANN + keyword matching via Reciprocal Rank Fusion (RRF)                     |
| **3. Route**      | Classifies query intent automatically                          | Factual / Semantic / Behavioral — RL bandit or rule-based fallback                          |
| **4. Retrieve**   | Searches KB with the best strategy for the query type          | Entity matching, keyword overlap, or frequency aggregation                                  |
| **5. Synthesize** | Produces a grounded answer with confidence grade and citations | Grade A–F, numeric score 0–1, source photo filenames                                        |

Three query modes:

- **Factual** — "How much did I spend at ALDI?" → extracts $18.69 from receipt OCR
- **Semantic** — "Show me photos of pizza" → matches against visual descriptions
- **Behavioral** — "What type of food do I photograph most?" → aggregates patterns across all photos

Every answer includes a confidence grade (A–F) and cites the specific source photo.

---

## The Stack

**AI / Models**

- **GPT-4o Vision** — photo analysis, OCR, entity extraction, classification
- **CrewAI** — multi-agent orchestration (4 agents, 2 crews)
- **sentence-transformers** — `all-MiniLM-L6-v2` for local embeddings (384-dim)
- **PyTorch** — DQN confidence calibrator
- **scikit-learn** — KMeans clustering for bandit context features

**Search & Storage**

- **Qdrant** — vector database with hybrid search (dense ANN + keyword via RRF)
- **JSON** — flat-file backend (default; also used for offline RL training)
- **Repository pattern** — swappable backends via ABC

**Backend & API**

- **FastAPI** — REST API with LRU cache, API key auth, SSE streaming, multi-user scoping
- **Pydantic** — settings management, input validation

**Frontend**

- **React** + **TypeScript** + **Vite** — SPA frontend
- **MUI** (Material UI) — component library
- **Recharts** — data visualization

---

## Architecture

<p align="center">
  <img src="docs/mermaid_diagrams/high_level_architecture.png" alt="High-Level Architecture" width="90%"/>
</p>

```
INGESTION CREW (Process.sequential)
  [Scan photos/] → [Analyze with GPT-4o Vision] → [Build JSON knowledge base]
  (--direct flag available for faster batch processing via direct API calls)
  Dual-write: ingests into both JSON file and Qdrant vector DB simultaneously

QUERY CREW (Process.hierarchical, manager-delegated)
  [Controller] classifies query intent → delegates to specialists
    ├── Task 1: Knowledge Retriever — searches KB with PhotoKnowledgeBaseTool
    └── Task 2: Insight Synthesizer — synthesizes answer with confidence grade + citation

STORAGE LAYER (Repository Pattern, selected by REPOSITORY_BACKEND)
  ├── JsonPhotoRepository (default)  — flat JSON file, also used for offline/RL training
  └── QdrantPhotoRepository          — vector DB with hybrid search (RRF)

API SERVER (FastAPI)
  uvicorn api.server:app — REST API with LRU cache, API key auth, multi-user scoping

FEEDBACK LOOP (persistent, adaptive)
  [Eval results] → [FeedbackStore] → adjusts confidence thresholds per strategy
```

### Agents

| Agent                   | Role                                                                        | Tools                                         |
| ----------------------- | --------------------------------------------------------------------------- | --------------------------------------------- |
| **Controller**          | Orchestrates query routing; classifies intent (factual/semantic/behavioral) | — (manager)                                   |
| **Photo Analyst**       | Analyzes images: OCR, entity extraction, classification                     | PhotoVisionTool, DirectoryReadTool            |
| **Knowledge Retriever** | Searches knowledge base; retrieves evidence                                 | PhotoKnowledgeBaseTool (custom), FileReadTool |
| **Insight Synthesizer** | Synthesizes grounded answers with confidence grading                        | FileReadTool, SerperDevTool (optional)        |

### Tools

| Tool                     | Type                | Purpose                                                                           |
| ------------------------ | ------------------- | --------------------------------------------------------------------------------- |
| `PhotoVisionTool`        | Custom              | GPT-4o Vision wrapper with HEIC support (`src/tools/photo_vision.py`)             |
| `PhotoKnowledgeBaseTool` | Custom              | 3-strategy search with feedback integration (`src/tools/photo_knowledge_base.py`) |
| `DirectoryReadTool`      | Built-in            | Scans the photos directory                                                        |
| `FileReadTool`           | Built-in            | Reads the knowledge base file                                                     |
| `JSONSearchTool`         | Built-in            | Embedding-based JSON search (sentence-transformers)                               |
| `SerperDevTool`          | Built-in (optional) | Web search enrichment                                                             |

---

## Setup

### Prerequisites

- Python 3.10–3.13 (CrewAI requires `< 3.14`)
- OpenAI API key (GPT-4o access)
- Docker (optional — only needed for the Qdrant backend)

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
- `API_KEY` — if set, POST endpoints on the API server require an `X-API-Key` header

### 4. Create directories and add photos

```bash
mkdir -p photos knowledge_base eval/results
```

Place 15–25 photos in `photos/`. iPhone photos (HEIC format) are fully supported. Recommended mix for best demo coverage:

- Grocery/restaurant receipts (3–5)
- Food photos (3–5)
- Screenshots from apps or websites (1–2)
- Documents or notes (1–2)

### 5. Start Qdrant (optional)

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Set `REPOSITORY_BACKEND=qdrant` in `.env` to use it. The ingestion pipeline auto-creates the `photos` collection on first write. If Qdrant is unavailable, the system falls back to the JSON backend automatically.

---

## Usage

### Ingest photos

```bash
# Default: CrewAI multi-agent pipeline
python -m src.main ingest

# Fast mode: direct API calls, bypasses CrewAI agents
python -m src.main ingest --direct
```

Analyzes all photos in `photos/` using GPT-4o Vision and writes them to `knowledge_base/photo_index.json` (and Qdrant, when enabled). Idempotent — re-running skips already-indexed photos.

### Query the knowledge base

```bash
python -m src.main query "How much did I spend at ALDI?"          # factual
python -m src.main query "Show me photos of pizza"                 # semantic
python -m src.main query "What type of food do I photograph most?" # behavioral
python -m src.main query "What was my electric bill?"              # edge case — should decline

# Fast path — skip CrewAI, zero OpenAI cost, sub-second latency
python -m src.main query --direct "How much did I spend at ALDI?"
```

### Run the API server

```bash
uvicorn api.server:app --reload --port 8000
```

The FastAPI server exposes the knowledge base and RL models as a REST API. Two query modes:

- **Fast** (`mode: "fast"`) — direct Python search with RL routing, no OpenAI calls, <1s, free
- **Full** (`mode: "full"`) — CrewAI pipeline with GPT-4o reasoning, 15–45s, ~$0.01–0.05/query

#### API Endpoints

| Endpoint                          | Method | Auth    | Description                                                                                            |
| --------------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------ |
| `/api/query`                      | POST   | API key | Query the knowledge base                                                                               |
| `/api/query/stream`               | POST   | API key | SSE streaming variant — emits `routing`/`retrieval`/`token`/`done` events for progressive UI rendering |
| `/api/feedback`                   | POST   | API key | Submit feedback for adaptive thresholds                                                                |
| `/api/cache/clear`                | POST   | API key | Clear the LRU query cache                                                                              |
| `/api/knowledge-base`             | GET    | —       | List all indexed photos                                                                                |
| `/api/knowledge-base/stats`       | GET    | —       | Knowledge base statistics                                                                              |
| `/api/photos/{photo_id}/thumbnail`| GET    | —       | Photo thumbnail                                                                                        |
| `/api/photos/{photo_id}/image`    | GET    | —       | Full-size photo                                                                                        |
| `/api/feedback/stats`             | GET    | —       | Per-strategy feedback statistics                                                                       |
| `/api/eval/results`               | GET    | —       | Latest evaluation results                                                                              |
| `/api/eval/ablation`              | GET    | —       | Ablation study results                                                                                 |
| `/api/eval/rl-training`           | GET    | —       | RL training results                                                                                    |
| `/api/eval/rl-eval`               | GET    | —       | RL evaluation results                                                                                  |
| `/api/figures`                    | GET    | —       | List generated figures                                                                                 |
| `/api/figures/{filename}`         | GET    | —       | Fetch a figure                                                                                         |
| `/api/system/architecture`        | GET    | —       | System architecture metadata                                                                           |
| `/api/health`                     | GET    | —       | Health check with backend status                                                                       |

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
python -m src.main eval                    # 20 original queries
python -m src.main eval --suite=expanded   # 83 queries
python -m src.main eval --suite=held_out   # 21 held-out queries
python -m src.main eval --suite=novel      # 15 intent-shift queries
```

Reports:

- Retrieval Accuracy — was the correct source photo found?
- Routing Accuracy — was the query intent correctly detected?
- Silent Failure Rate — did the system ever confidently return a wrong answer?
- Decline Accuracy — were impossible queries correctly refused?

Results saved to `eval/results/eval_results.json`. Run history appended to `eval/results/eval_history.json` for trend analysis.

### Train the RL components (offline, no API calls)

```bash
# Train both components across 5 seeds (4000 episodes each, ~120s total)
python -m src.main train

# Train with custom episode count
python -m src.main train 1000

# Run full RL evaluation: 5 configs × 5 seeds × 83 queries
python -m src.main rl-eval

# Run 7-config ablation study → eval/results/ablation_results.json
python -m src.main ablation
```

RL training runs entirely on the cached knowledge base (zero API cost). Requires an existing `knowledge_base/photo_index.json` — run ingestion first. Trained models are saved to `knowledge_base/rl_models/` and loaded automatically at query time.

---

## Reinforcement Learning Extension

PhotoMind integrates two RL approaches that replace rule-based components.

### Approach 1: Contextual Bandits — Query Routing

Replaces the keyword-based `_classify_query()` in `PhotoKnowledgeBaseTool` with a learned policy that selects the search strategy (factual / semantic / behavioral) based on query features.

- **ThompsonSamplingBandit** — Beta posterior per context cluster
- **UCBBandit** — UCB1 upper confidence bound per cluster
- **EpsilonGreedyBandit** — baseline comparison
- Context clustering via KMeans on 396-dimensional hybrid query feature vectors (12 handcrafted + 384 MiniLM embedding dims)
- Training: 4000 episodes × 5 seeds on offline cached search results (zero API cost)

### Approach 2: DQN — Confidence Calibration

Replaces static confidence thresholding with a DQN that learns when to accept, hedge, or decline retrieval results — directly addressing the silent failure problem.

- Architecture: FC(8→64) → ReLU → FC(64→64) → ReLU → FC(64→5), adapted from the LunarLander DQN (extended with a requery action)
- State: 8-dim vector (top score, score gap, result count, strategy index, query features, entity match indicators)
- Actions: `accept_high`, `accept_moderate`, `hedge`, `requery`, `decline`
- Reward: penalty matrix that heavily penalizes silent failures (confident-but-wrong answers)

Key result: **every RL config lowers the silent failure rate** relative to the rule-based baseline (4.8% → 0.2–3.4%). See [Evaluation Results](#evaluation-results).

### RL Architecture

<p align="center">
  <img src="docs/mermaid_diagrams/system_Architecture_with_RL.png" alt="System Architecture with RL" width="90%"/>
</p>

**Offline simulation training:** Both components are trained using `PhotoMindSimulator`, which pre-computes all 3 search strategies on all 83 queries once (zero API calls). Training 4000 episodes × 5 seeds × 2 components takes ~120 seconds on CPU.

---

## Custom Tool: PhotoKnowledgeBaseTool

`src/tools/photo_knowledge_base.py` — the core differentiator.

**Three search strategies selected automatically by query intent:**

| Strategy       | Trigger Keywords                                    | How It Works                                                    |
| -------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| **Factual**    | "how much", "date", "address", "items", "vendor"    | Entity matching + OCR text search                               |
| **Semantic**   | Visual/descriptive words, or no other match         | Keyword overlap on descriptions, normalized by meaningful words |
| **Behavioral** | "most", "often", "how many", "breakdown", "pattern" | Frequency aggregation across all photos                         |

**Output always includes:** confidence grade (A–F), numeric score (0–1), source photo filenames, and a plain-language summary.

**Input schema (Pydantic):**

```python
query: str                          # Natural language question
query_type: str = "auto"            # Force strategy or let tool classify
top_k: int = 3                      # Number of results
confidence_threshold: float = 0.25  # Minimum score to include
```

---

## Evaluation Results

### Base System (53 photos, 20 queries)

Latest `python -m src.main eval` run (JSON backend, CrewAI pipeline):

| Metric              | Score      |
| ------------------- | ---------- |
| Retrieval Accuracy  | **90%**    |
| Routing Accuracy    | **95%**    |
| Silent Failure Rate | **5%**     |
| Decline Accuracy    | **100%**   |
| Avg Latency         | ~37s/query |

### RL Extension (53 photos, 83 queries, 5 seeds)

From `eval/results/rl_eval_results.json` (mean over 5 seeds; baseline is deterministic):

| Config                 | Retrieval | Routing   | Silent Fail | Decline Acc |
| ---------------------- | --------- | --------- | ----------- | ----------- |
| Baseline (rule-based)  | 85.5%     | **75.9%** | 4.8%        | **92.9%**   |
| Bandit Only (Thompson) | **90.1%** | 65.8%     | **0.2%**    | 91.4%       |
| DQN Only               | 85.5%     | **75.9%** | 3.4%        | 84.3%       |
| Full RL (Thompson+DQN) | 89.6%     | 66.5%     | 1.7%        | 90.0%       |
| Recommended (Rule+DQN) | 85.5%     | **75.9%** | 3.4%        | 84.3%       |

<p align="center">
  <img src="docs/figures/ablation_comparison.png" alt="Ablation Study Comparison" width="90%"/>
</p>

---

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
│   │   ├── embedding_index.py       # Local embedding index (MiniLM)
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
│   ├── expanded_test_cases.py       # 63 new cases (incl. 14 ambiguous) — 83 total
│   ├── novel_test_cases.py          # 15 intent-shift queries for robustness testing
│   ├── run_evaluation.py            # Base system evaluation harness
│   ├── run_rl_evaluation.py         # RL 5-config comparison harness
│   ├── ablation.py                  # 7-config ablation with paired t-tests
│   ├── statistical_analysis.py      # CI, paired t-test, Cohen's d utilities
│   ├── run_e2e_production.py        # E2E tests (factual + edge) against the deployed API
│   ├── e2e_behavioral_novel.py      # E2E tests (behavioral + novel) against the deployed API
│   ├── run_browser_sweep.py         # 75-query sweep against the deployed API
│   ├── submit_feedback.py           # Submit judge-labeled feedback to the deployed API
│   └── results/                     # JSON results + eval history
├── viz/
│   ├── plot_learning_curves.py      # Bandit regret, DQN rewards, posteriors, epsilon decay
│   ├── plot_ablation.py             # Grouped bar chart (7 configs × 4 metrics)
│   ├── plot_regret.py               # Cumulative regret comparison (3 bandit types)
│   ├── plot_before_after.py         # Before/after RL comparison plots
│   ├── generate_diagrams.py         # Architecture and flow diagrams
│   ├── figures/                     # Generated PNG and PDF figures
│   └── diagrams/                    # Mermaid diagram sources (.mmd)
├── scripts/
│   ├── train_full.py                # Full training + optional ablation
│   ├── train_bandit.py              # Standalone bandit training
│   ├── train_dqn.py                 # Standalone DQN training
│   ├── precompute_cache.py          # Pre-compute search strategy cache
│   ├── scaling_benchmark.py         # Scaling and performance benchmarks
│   ├── migrate_to_qdrant.py         # JSON → Qdrant migration utility
│   ├── fetch_assets.py              # Fetch photos + KB from R2 at container start
│   └── demo_comparison.py           # Rule-based vs RL before/after demo
├── knowledge_base/
│   ├── photo_index.json             # 53 indexed photos
│   ├── feedback_store.json          # Persistent feedback-loop state
│   └── rl_models/                   # Trained RL models
│       ├── bandit_thompson.pkl      # Trained Thompson Sampling bandit
│       └── dqn_confidence.pth       # Trained DQN confidence calibrator
├── tests/
│   ├── test_core.py                 # Core RL functionality tests (59 tests)
│   ├── test_search_strategies.py    # Search strategy correctness tests (24 tests)
│   └── test_repository.py           # Repository abstraction tests (13 tests)
├── web/                             # React + TypeScript + Vite frontend (MUI, Recharts)
│   ├── src/
│   │   ├── App.tsx                  # Main app component
│   │   ├── components/              # UI components
│   │   └── theme.ts                 # MUI theme configuration
│   ├── index.html
│   └── package.json
├── docs/
│   ├── index.html                   # GitHub Pages site
│   ├── figures/                     # Copied figures for GitHub Pages
│   ├── math_formulations.md         # Mathematical formulations for RL components
│   ├── mermaid_diagrams/            # Rendered architecture diagrams (PNG)
│   └── report/                      # LaTeX technical report (main.tex → main.pdf)
├── Dockerfile                       # Multi-stage build (Node frontend + Python backend)
├── Dockerfile.render                # Render deployment variant
├── .dockerignore
├── .env.example
├── .gitignore
├── LICENSE
├── requirements.txt
├── TECHNICAL_REPORT.md              # Full technical documentation (base system + RL extension)
└── TECHNICAL_REPORT.pdf
```

---

## License

MIT License

_This is an academic project demonstrating multimodal AI retrieval with reinforcement learning. Not intended for production use._
