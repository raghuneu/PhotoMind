# PhotoMind

A multimodal personal photo knowledge retrieval system built with CrewAI. Turns your phone's photo library into a queryable knowledge base — ask natural-language questions, get answers with confidence scores and source attribution.

[![Demo Video](Photomind.png)](https://youtu.be/wcw8_X2_HGE)

## What It Does

PhotoMind uses GPT-4o Vision to analyze personal photos (bills, receipts, food photos, screenshots, documents) and builds a searchable JSON knowledge base. Three query modes:

- **Factual** — "How much did I spend at ALDI?" → extracts $18.69 from receipt OCR
- **Semantic** — "Show me photos of pizza" → matches against visual descriptions
- **Behavioral** — "What type of food do I photograph most?" → aggregates patterns across all photos

Every answer includes a confidence grade (A–F) and cites the specific source photo.

## Architecture

```
INGESTION CREW (Process.sequential)
  [Scan photos/] → [Analyze with GPT-4o Vision] → [Build JSON knowledge base]
  (--direct flag available for faster batch processing via direct API calls)

QUERY CREW (Process.hierarchical, manager-delegated)
  [Controller] classifies query intent → delegates to specialists
    ├── Task 1: Knowledge Retriever — searches KB with PhotoKnowledgeBaseTool
    └── Task 2: Insight Synthesizer — synthesizes answer with confidence grade + citation

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

### 4. Create directories and add photos

```bash
mkdir -p photos knowledge_base eval/results
```

Place 15–25 photos in `photos/`. iPhone photos (HEIC format) are fully supported. Recommended mix for best demo coverage:
- Grocery/restaurant receipts (3–5)
- Food photos (3–5)
- Screenshots from apps or websites (1–2)
- Documents or notes (1–2)

## Usage

### Ingest photos

```bash
# Default: CrewAI multi-agent pipeline
python -m src.main ingest

# Fast mode: direct API calls, bypasses CrewAI agents
python -m src.main ingest --direct
```

Analyzes all photos in `photos/` using GPT-4o Vision and writes `knowledge_base/photo_index.json`. Idempotent — re-running skips already-indexed photos. The default mode uses CrewAI agents for orchestrated ingestion; `--direct` is faster for batch processing.

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

## Evaluation Results (25 photos, 20 queries)

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | **95%** |
| Routing Accuracy | **100%** |
| Silent Failure Rate | **5%** |
| Decline Accuracy | **100%** |
| Avg Latency | ~30s/query |

## Project Structure

```
PhotoMind/
├── src/
│   ├── main.py                      # CLI entry point (ingest / query / eval)
│   ├── config.py                    # Pydantic settings (reads .env)
│   ├── ingest_direct.py             # Direct ingestion (1 API call/photo)
│   ├── agents/
│   │   └── definitions.py           # 4 agent factory functions
│   ├── tasks/
│   │   ├── ingestion.py             # Scan, analyze, index tasks
│   │   └── query.py                 # Query task with intent routing
│   ├── crews/
│   │   ├── ingestion_crew.py        # Sequential ingestion pipeline
│   │   └── query_crew.py            # Hierarchical query pipeline
│   └── tools/
│       ├── photo_vision.py          # PhotoVisionTool (GPT-4o Vision + HEIC)
│       ├── photo_knowledge_base.py  # PhotoKnowledgeBaseTool (custom)
│       └── feedback_store.py        # FeedbackStore (adaptive threshold learning)
├── eval/
│   ├── test_cases.py                # 20 hand-labeled test queries
│   ├── run_evaluation.py            # Metrics harness
│   └── results/
│       └── eval_results.json        # Latest eval output
├── knowledge_base/
│   └── photo_index.json             # Generated — 25 photos indexed
├── photos/                          # Your images (JPG, PNG, HEIC)
├── requirements.txt
├── .env.example
├── TECHNICAL_REPORT.md              # Full technical documentation
└── PROJECT_RETROSPECTIVE.md         # Issue log with root causes and fixes
```

## Known Limitations

- Semantic search uses keyword overlap, not true vector embeddings — misses synonyms
- Knowledge base is a flat JSON file — suitable up to ~500 photos; use a vector DB beyond that
- Confidence grading is calibrated for a small corpus — thresholds may need tuning at scale
- Query routing is rule-based; edge cases may misclassify unusual phrasings
