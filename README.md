# PhotoMind

> **Take-Home Final: Reinforcement Learning for Agentic AI Systems**
> This project is the submission for the take-home final. The work lives on the `feature/reinforcement-learning-extension` branch

A multimodal personal photo knowledge retrieval system built with CrewAI. Turns your phone's photo library into a queryable knowledge base — ask natural-language questions, get answers with confidence scores and source attribution.

**Repository:** [github.com/raghuneu/PhotoMind](https://github.com/raghuneu/PhotoMind/tree/feature/reinforcement-learning-extension) · **Demo:** [youtu.be/UQRdkW2mAgc](https://www.youtube.com/watch?v=UQRdkW2mAgc)

[![Demo Video](Photomind_withRL.png)](https://www.youtube.com/watch?v=UQRdkW2mAgc)

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

### Train the RL components (offline, no API calls)

```bash
# Train both components across 5 seeds (2000 episodes each, ~60s total)
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
- Context clustering via KMeans on 12-dimensional query feature vectors
- Training: 2000 episodes × 5 seeds on offline cached search results (zero API cost)

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
[QueryFeatureExtractor]  →  12-dim feature vector
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

**Offline simulation training:** Both components are trained using `PhotoMindSimulator`, which pre-computes all 3 search strategies on all 56 queries once (zero API calls). Training 2000 episodes × 5 seeds × 2 components takes ~60 seconds on CPU.

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

### Base System (25 photos, 20 queries)

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | **95%** |
| Routing Accuracy | **100%** |
| Silent Failure Rate | **5%** |
| Decline Accuracy | **100%** |
| Avg Latency | ~30s/query |

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
├── src/
│   ├── main.py                      # CLI entry point (ingest / query / eval / train / rl-eval / ablation)
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
│   ├── tools/
│   │   ├── photo_vision.py          # PhotoVisionTool (GPT-4o Vision + HEIC)
│   │   ├── photo_knowledge_base.py  # PhotoKnowledgeBaseTool (custom) — RL-enhanced
│   │   ├── query_memory.py          # Query memory and deduplication
│   │   └── feedback_store.py        # FeedbackStore (adaptive threshold learning)
│   └── rl/
│       ├── rl_config.py             # Centralized RL hyperparameters and reward matrix
│       ├── feature_extractor.py     # Query → 12-dim feature vector
│       ├── contextual_bandit.py     # Thompson Sampling, UCB, epsilon-greedy bandits
│       ├── dqn_confidence.py        # ConfidenceDQN and ConfidenceDQNAgent
│       ├── replay_buffer.py         # Experience replay buffer (adapted from LunarLander)
│       ├── reward.py                # Reward computation for bandit and DQN
│       ├── simulation_env.py        # Offline training environment (zero API cost)
│       └── training_pipeline.py     # Orchestrates training across seeds
├── eval/
│   ├── test_cases.py                # 20 original hand-labeled test queries
│   ├── expanded_test_cases.py       # 36 new cases (incl. 11 ambiguous) — 56 total
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
│   └── test_core.py                 # Core functionality tests
├── docs/
│   ├── math_formulations.md         # Mathematical formulations for RL components
│   └── mermaid_diagrams/            # Mermaid diagram sources
├── .env.example
├── .gitignore
├── LICENSE
├── PROJECT_RETROSPECTIVE.md         # Project retrospective and lessons learned
├── requirements.txt
└── TECHNICAL_REPORT.md              # Full technical documentation (base system + RL extension)
```

## Known Limitations

- Semantic search uses keyword overlap, not true vector embeddings — misses synonyms
- Knowledge base is a flat JSON file — suitable up to ~500 photos; use a vector DB beyond that
- Confidence grading is calibrated for a small corpus — thresholds may need tuning at scale
- RL bandit trained on 56 queries with 10x augmentation — may not generalize to unseen phrasing patterns outside the training distribution
- DQN requery action selects an alternate strategy randomly rather than learning which alternate to try; a learned requery policy could improve multi-step episode returns
- Bandit context clustering uses k=4 clusters on a small feature space — more data would support finer-grained contextualization
