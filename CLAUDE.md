# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhotoMind is a multimodal personal photo knowledge retrieval system built with CrewAI. It uses GPT-4o Vision to analyze personal photos (bills, receipts, food, screenshots, scenes) and builds a searchable JSON knowledge base supporting natural-language queries with confidence scoring and source attribution.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys (OPENAI_API_KEY required, SERPER_API_KEY optional)
cp .env.example .env

# Create required directories
mkdir -p photos knowledge_base eval/results

# Ingest photos into knowledge base (CrewAI agents)
python -m src.main ingest

# Ingest photos (direct mode — faster, bypasses CrewAI agents)
python -m src.main ingest --direct

# Query the knowledge base
python -m src.main query "How much did I spend at ALDI?"

# Run evaluation suite (20 test queries, results → eval/results/eval_results.json)
python -m src.main eval

# Train RL components (offline — no API calls; ~120s on CPU for 4000 eps x 5 seeds)
python -m src.main train

# Run RL evaluation: 4 configs x 5 seeds x 56 queries → eval/results/rl_eval_results.json
python -m src.main rl-eval

# Run 7-config ablation study → eval/results/ablation_results.json
python -m src.main ablation
```

No test framework (pytest, etc.) — evaluation is the test suite via `python -m src.main eval`. Unit tests exist in `tests/` (96 tests across 3 files: test_core.py, test_search_strategies.py, test_repository.py) and run via `pytest tests/ -v`.

RL training requires no API keys; trained models are saved to `knowledge_base/rl_models/` and
loaded automatically by `PhotoKnowledgeBaseTool` at query time (with fallback to rule-based).

## Architecture

Two CrewAI pipelines with four agents:

**Ingestion Crew** (`src/crews/ingestion_crew.py`) — `Process.sequential`

- Scan `photos/` → Analyze each photo with GPT-4o Vision → Build/update `knowledge_base/photo_index.json`
- Agents used: Photo Analyst, Knowledge Retriever
- Ingestion is idempotent: re-running skips already-indexed photos
- Fallback: `--direct` flag bypasses CrewAI for faster batch processing via `src/ingest_direct.py`

**Query Crew** (`src/crews/query_crew.py`) — `Process.hierarchical` with manager delegation

- Controller agent classifies query intent (factual/semantic/behavioral), delegates to specialists
- Two-task pipeline: Knowledge Retriever searches the KB, then Insight Synthesizer produces the final graded answer
- Agents used: Controller (manager), Knowledge Retriever, Insight Synthesizer
- Returns structured answer with confidence grade (A-F) and source photo citations

**Agent definitions** live in `src/agents/definitions.py` as factory functions. Each returns a `crewai.Agent` with preconfigured role, goal, backstory, and tools.

## Custom Tool: PhotoKnowledgeBaseTool

`src/tools/photo_knowledge_base.py` — the core differentiator. Implements three search strategies selected by auto-detected query intent:

- **Factual**: entity matching + OCR text search (for bills, receipts, specific facts)
- **Semantic**: keyword overlap on descriptions/captions (for visual similarity, mood)
- **Behavioral**: query-aware frequency aggregation across photo corpus (for patterns, habits)

Outputs include: confidence grade (A-F), numeric score (0-1), source photo filenames, strategy accuracy from feedback loop, and warnings for low-confidence results.

## Feedback Loop

`src/tools/feedback_store.py` implements a persistent feedback loop backed by `knowledge_base/feedback_store.json`. After each eval run, per-strategy accuracy rates are computed and used to adaptively adjust confidence thresholds:

- Strategy accuracy < 70% → threshold increased by +0.05 (more conservative)
- Strategy accuracy >= 90% → threshold decreased by -0.05 (less conservative)

Eval history is tracked in `eval/results/eval_history.json` for trend analysis.

## Key Design Decisions

- **Knowledge base format**: Flat JSON file (`photo_index.json`) with metadata header and photos array. Each photo record has: id, file_path, filename, image_type, ocr_text, description, entities, confidence, indexed_at.
- **Query intent routing**: Rule-based keyword classification in `PhotoKnowledgeBaseTool._classify_query()`, not LLM-based. Checks behavioral → factual → semantic keywords in priority order.
- **Confidence grading**: A (>=0.7), B (>=0.5), C (>=0.35), D (>=0.2), F (<0.2). Calibrated for keyword-based retrieval. Grades A/B/C are treated as strong; D/F trigger warnings.
- **Config**: `src/config.py` uses Pydantic Settings with `.env` file. Fails fast if `OPENAI_API_KEY` is missing.

## RL Extension

`src/rl/` contains two RL components integrated into `PhotoKnowledgeBaseTool`:

**RL Approach 1 — Contextual Bandits** (`src/rl/contextual_bandit.py`)
- Replaces rule-based `_classify_query()` with a learned routing policy
- Three algorithms: `ThompsonSamplingBandit`, `UCBBandit`, `EpsilonGreedyBandit`
- Context: KMeans clusters (k=4) on 396-dim hybrid query feature vectors (12 handcrafted + 384 MiniLM embedding dims) (`feature_extractor.py`)
- Trained offline via `PhotoMindSimulator` — zero API calls

**RL Approach 2 — DQN Confidence Calibrator** (`src/rl/dqn_confidence.py`)
- Replaces static `_score_to_grade()` with a learned accept/hedge/decline policy
- Architecture: FC(8→64) → ReLU → FC(64→64) → ReLU → FC(64→5) — adapted from LunarLander DQN (extended with requery action)
- State dim=8: top score, score gap, result count, strategy index, query features, entity match indicators
- Actions: `accept_high`, `accept_moderate`, `hedge`, `requery`, `decline`
- Reward: outcome-based aggregate with `expected_top_entity`; penalizes silent failures (confident-but-wrong) at -1.0

**Integration:** `PhotoKnowledgeBaseTool._rl_classify_query()` and `_rl_confidence_grade()` try RL first, fall back to rule-based if models not found. Trained models at `knowledge_base/rl_models/`.

**Training pipeline:** `src/rl/training_pipeline.py` orchestrates multi-seed training + ablation. The `PhotoMindSimulator` pre-computes all 3 search strategies on all queries once, enabling thousands of training episodes with no API calls.

## Evaluation

`eval/test_cases.py` has 20 hand-labeled test queries across 4 categories: factual (7), semantic (5), behavioral (4), edge cases (4). Edge cases have `should_decline: True` — the system should gracefully refuse rather than hallucinate.

`eval/expanded_test_cases.py` adds 36 more queries (7 factual, 6 semantic, 6 behavioral, 6 edge cases, 11 ambiguous) for a total of 56 queries (`ALL_TEST_CASES`). The 11 ambiguous queries are critical for demonstrating RL value — they expose cases where keyword routing systematically fails.

`eval/novel_test_cases.py` adds 15 intent-shift queries that probe robustness to mid-sentence category switches. Run via `python -m src.main eval --suite=novel`.

`eval/run_evaluation.py` measures: retrieval accuracy, routing accuracy, silent failure rate, decline accuracy, and per-query latency.

`eval/run_rl_evaluation.py` compares 4 configs (baseline / bandit-only / DQN-only / full RL) with 95% CIs and paired t-tests.

`eval/ablation.py` runs 7-config ablation study with Cohen's d effect sizes.

## Tools Used by Agents

3 built-in CrewAI tools + 2 custom:

- `PhotoVisionTool` — Custom GPT-4o Vision wrapper with HEIC support (`src/tools/photo_vision.py`)
- `PhotoKnowledgeBaseTool` — Custom multi-strategy search with feedback integration (`src/tools/photo_knowledge_base.py`)
- `DirectoryReadTool` — Built-in photo directory scanning
- `FileReadTool` — Built-in KB file access
- `JSONSearchTool` — Built-in embedding-based JSON search (sentence-transformers)

Optional: `SerperDevTool` for web search enrichment (requires `SERPER_API_KEY`).

# Additional Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Security Awareness

- Never hardcode API keys or secrets in source code — use `.env` and `src/config.py`.
- Verify `.env` is in `.gitignore` before any commit.
- `knowledge_base/photo_index.json` contains personal photo descriptions — treat as sensitive data.
- Error messages must not leak API keys, file paths, or personal content to stdout in production contexts.

## 6. Git Commit Convention

Use `<type>: <description>` format for commit messages:

- `feat`: new feature or capability
- `fix`: bug fix
- `refactor`: code restructure without behavior change
- `docs`: documentation only
- `test`: evaluation or test changes
- `chore`: dependency updates, config, CI
- `perf`: performance improvement

## 7. Code Quality Checklist

Before marking work complete, verify:

- [ ] Functions are focused (< 50 lines)
- [ ] Files are manageable (< 800 lines)
- [ ] No magic numbers — use named constants or config values
- [ ] No deep nesting (> 4 levels) — prefer early returns
- [ ] Error handling at system boundaries (user input, API calls, file I/O)
- [ ] Existing `python -m src.main eval` still passes after changes

---
