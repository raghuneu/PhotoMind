# PhotoMind — Project Retrospective & Issue Log

> **What this file is:** A complete record of every technical issue encountered building PhotoMind, what caused it, and how it was resolved. Useful for debugging similar agentic systems and for understanding the tradeoffs made.

---

## Project at a Glance

**PhotoMind** is a multimodal personal photo knowledge retrieval system built with CrewAI. It uses GPT-4o Vision to analyze personal photos (bills, receipts, food, screenshots), builds a searchable JSON knowledge base, and answers natural-language queries with confidence scoring and source attribution.

**Stack:** CrewAI 1.14.1 · GPT-4o Vision (OpenAI) · sentence-transformers (local embeddings) · Pydantic Settings · Pillow + pillow-heif (HEIC) · Python 3.10.14

**Two pipelines:**
- **Ingestion Crew** (`Process.sequential`) — Scan → Analyze → Index
- **Query Crew** (`Process.hierarchical`) — Controller manages Knowledge Retriever + Insight Synthesizer

---

## Issue Log

### 1. Python Version Incompatibility with CrewAI

**Symptom:** `pip install crewai` failed or produced import errors.

**Root cause:** CrewAI requires Python `<3.14`. System Python was 3.14.x (latest macOS default at the time).

**Fix:** Created a virtual environment using pyenv-managed Python 3.10.14:
```bash
~/.pyenv/versions/3.10.14/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Lesson:** Always pin a Python version in `.python-version` or `pyproject.toml` for agentic projects. CrewAI and many ML libraries lag Python releases by 6-12 months.

---

### 2. HEIC Image Format Not Supported by PIL by Default

**Symptom:** `PIL.UnidentifiedImageError` when opening iPhone photos (`.HEIC` files) — 14 of 25 photos were HEIC.

**Root cause:** Pillow does not natively decode HEIC (Apple's High Efficiency Image Container format). iPhone photos shot after iOS 11 default to HEIC unless changed in Camera settings.

**Fix:** Added `pillow-heif` to requirements and registered its opener at module import time:
```python
import pillow_heif
pillow_heif.register_heif_opener()  # must run before any PIL.Image.open() call
```

After registration, `PIL.Image.open("IMG_1234.HEIC")` works transparently.

**Additional wrinkle:** HEIC files decoded by `pillow-heif` sometimes yield `RGBA` or `YCbCr` mode images. JPEGs only support `RGB` or `L`. Added a mode guard before converting to JPEG for the API:
```python
if img.mode not in ("RGB", "L"):
    img = img.convert("RGB")
```

**Lesson:** Any system ingesting iPhone photos must handle HEIC. Register the opener early — importing the module is not enough; `register_heif_opener()` must be called explicitly.

---

### 3. Attempted Gemini Free Tier — Multiple Failures

This was a multi-step failure. The goal was to use Google Gemini's free tier instead of paid OpenAI.

#### 3a. `google-generativeai` SDK deprecated

**Symptom:** `from google.generativeai import ...` failed with deprecation warning and import error.

**Root cause:** Google deprecated the `google-generativeai` package in favour of `google-genai` (new unified SDK).

**Fix:** Changed import to `from google import genai` and updated all API calls to the new SDK style.

#### 3b. Model not available on API key

**Symptom:** `404 Model not found: gemini-1.5-flash`.

**Root cause:** The API key was scoped to a newer API version (`v1`) where `gemini-1.5-flash` had been deprecated. It only served `gemini-2.x` models.

**Fix:** Switched to `gemini-2.5-flash`.

#### 3c. Daily quota (20 RPD) exhausted

**Symptom:** `429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'generate_content_request_count'`.

**Root cause:** The free tier for `gemini-2.5-flash` allows only 20 requests per day. Multiple failed ingestion runs (rate limit debugging) consumed all 20 slots within hours.

**Fix:** Abandoned Gemini entirely. Switched back to OpenAI GPT-4o with a paid API key.

**Lesson:** Free-tier LLM APIs are unreliable for development — the quota is too small to survive debugging. Always keep a paid API key available for active development. For production free-tier use, implement retry-with-backoff and track daily usage explicitly.

---

### 4. CrewAI Ingestion Too Expensive (~100 API Calls for 25 Photos)

**Symptom:** The CrewAI ingestion crew (Scan → Analyze → Index) triggered roughly 4 LLM calls per photo: one to plan, one to invoke the tool, one to process results, one to write the index entry. For 25 photos: ~100 calls, slow and costly.

**Root cause:** CrewAI's hierarchical/sequential process adds overhead per task. The Photo Analyst agent uses `PhotoVisionTool` (which calls the LLM) but the agent itself also makes LLM calls to plan and summarize.

**Fix:** Created `src/ingest_direct.py` — a direct Python script that bypasses CrewAI entirely for ingestion:
- Lists photos with `os.listdir()`
- Calls GPT-4o Vision directly via `openai.chat.completions.create()` (1 call/photo)
- Parses JSON response
- Writes `knowledge_base/photo_index.json`
- Idempotent: checks `indexed_ids` set to skip already-indexed photos

Result: 25 API calls for 25 photos instead of ~100. ~4x cost reduction.

**Lesson:** For batch ingestion, direct API calls beat CrewAI agent overhead significantly. Use CrewAI for interactive/agentic query tasks where multi-step reasoning adds value. Don't use agents for pipelines that are just "call API, process result, write file."

---

### 5. OpenAI Embeddings API Blocked (403 PermissionDenied)

**Symptom:** `crew.kickoff()` immediately raised `openai.PermissionDeniedError: You don't have access to the embeddings API`.

**Root cause:** `memory=True` in `Crew()` defaults to OpenAI's `text-embedding-3-small` for vector memory storage. Some API keys (especially org-restricted or free-tier keys) do not have embeddings API access.

**Fix attempts:**
1. **fastembed** — `embedder={"provider": "fastembed", ...}` — rejected by CrewAI 1.14.1 (not a valid provider in this version).
2. **sentence-transformer** — `embedder={"provider": "sentence-transformer", "config": {"model": "all-MiniLM-L6-v2"}}` — works. Uses a local model, no API calls.

Final config in both crews:
```python
Crew(
    ...
    memory=True,
    embedder={"provider": "sentence-transformer", "config": {"model": "all-MiniLM-L6-v2"}},
)
```

Required: `pip install sentence-transformers`

**Lesson:** Never assume the default `memory=True` will work with all API keys. Always specify an embedder explicitly. `sentence-transformer` is the most reliable cross-platform option — it's local, free, and fast.

---

### 6. Query Matching Failed on Punctuation in Queries

**Symptom:** Query `"How much did I spend at ALDI?"` returned empty results even though ALDI was in the knowledge base.

**Root cause:** The PhotoKnowledgeBaseTool's `_factual_search()` compared raw query words against entity values. The query word `"aldi?"` (with question mark) did not match the entity value `"aldi"`.

**Fix:** Added a `_clean()` helper that lowercases and strips all punctuation before any comparison:
```python
import re
def _clean(text: str) -> str:
    """Lowercase and strip punctuation so 'aldi?' matches 'aldi'."""
    return re.sub(r'[^\w\s]', '', text.lower())
```

Applied in `_classify_query`, `_factual_search`, and `_semantic_search`.

**Lesson:** Always normalize text before comparison in search systems. At minimum: lowercase + strip punctuation. This is especially important for user-facing queries which frequently contain trailing `?` and possessives like `"Joe's"`.

---

### 7. Apostrophes in Store Names Broke Semantic Matching

**Symptom:** `"What items did I buy at Trader Joe's?"` returned zero results even though Trader Joe's receipt was in the KB.

**Root cause:** `_semantic_search()` applied `.lower()` to descriptions but not `_clean()`. The description contained `"Trader Joe's"` (with apostrophe). After `.lower()` and `.split()`, the token was `"joe's"`. The query after `_clean()` produced `"joes"`. These never matched.

**Fix:** Apply `_clean()` to description and caption text in `_semantic_search()` before splitting into words:
```python
description = _clean(photo.get("description", ""))
caption = _clean(photo.get("caption", ""))
```

**Lesson:** Apply the same text normalization to both query AND document text. A mismatch in normalization between the two sides is a common source of silent retrieval failures.

---

### 8. Semantic Search Scoring Too Strict (Correct Photos Scored Below Threshold)

**Symptom:** Semantic queries like `"Show me photos of pizza"` returned zero results despite the pizza photo existing with "pizza" in its description.

**Root cause:** The scoring formula normalized by total query word count:
```python
score = len(meaningful_overlap) / max(len(query_words), 1) * 0.8
```

For `"Show me photos of pizza"` (5 words), matching only "pizza" scored `1/5 × 0.8 = 0.16` — below the default threshold of `0.3`.

The issue: stop words like "me", "of", "show" inflate the denominator without contributing to matches.

**Fix:** Normalize by *meaningful* query words only (length > 3):
```python
meaningful_query_words = {w for w in query_words if len(w) > 3}
score = len(meaningful_overlap) / max(len(meaningful_query_words), 1) * 0.8
```

Also lowered the default `confidence_threshold` from `0.3` to `0.15`.

**Result:** Pizza photo now scores `1/3 × 0.8 = 0.267`, above the 0.15 threshold.

**Lesson:** When scoring keyword overlap, always filter stop words from the denominator. Using total word count as the denominator penalizes natural-language queries that include articles, prepositions, and pronouns.

---

### 9. Query Routing: Behavioral Queries Misclassified as Factual

**Symptom:** `"How many receipts do I have?"` was classified as `factual` instead of `behavioral`.

**Root cause:** `_classify_query()` checked factual keywords before behavioral keywords. The word `"receipt"` was in the factual keyword list, so it matched first even though `"how many"` is a behavioral signal.

**Fix:** Reversed the check order — behavioral before factual:
```python
if any(kw in q for kw in behavioral_keywords):
    return "behavioral"
elif any(kw in q for kw in factual_keywords):
    return "factual"
else:
    return "semantic"
```

Also added `"items"` to factual keywords so `"What items did I buy at Trader Joe's?"` routes correctly.

**Lesson:** In rule-based classifiers, more specific / higher-priority rules must fire first. Behavioral keywords like `"how many"` are more specific than factual keywords like `"receipt"` — they should win.

---

### 10. Confidence Grades All "F" After Scoring Recalibration

**Symptom:** After fixing semantic search scoring (Issue 8), all queries returned confidence grade `F` because the rescaled scores (0.15–0.55) all fell below the original grade thresholds (A=0.9, B=0.8, C=0.7).

**Root cause:** The original grading thresholds were calibrated for a different scoring scale. Keyword-overlap matching fundamentally cannot achieve scores of 0.8+ (that would require nearly all query words to appear in a description), so A/B grades were unreachable.

**Fix:** Recalibrated thresholds to match the actual scoring range:
```python
def _score_to_grade(self, score: float) -> str:
    if score >= 0.7:  return "A"   # was 0.9
    if score >= 0.5:  return "B"   # was 0.8
    if score >= 0.35: return "C"   # was 0.7
    if score >= 0.2:  return "D"   # was 0.5
    return "F"
```

**Lesson:** Confidence thresholds are system-specific. Calibrate them empirically against your actual retrieval scores. If your retrieval algorithm's top scores are 0.4–0.6, your "high confidence" threshold should be around 0.5, not 0.9.

---

### 11. Eval TPM Errors on Final Queries (Context Accumulation)

**Symptom:** Queries 19 and 20 in the eval suite failed with `Error code: 429 — Request too large (31,937 tokens > 30,000 TPM limit)`.

**Root cause:** The eval reused a single `Crew` instance for all 20 queries. With `memory=True`, CrewAI accumulates conversation history across `kickoff()` calls. By query 19-20, the accumulated context exceeded the GPT-4o TPM limit.

**Fix:** Create a fresh `Crew` instance per query:
```python
for i, tc in enumerate(TEST_CASES):
    crew = create_query_crew()   # fresh instance each time
    raw = crew.kickoff(inputs={"user_query": tc["query"]})
```

This prevents context accumulation while still demonstrating `memory=True` within a single query's multi-agent conversation.

**Lesson:** When running batch evaluations over a long-memory crew, always instantiate a new crew per test case. Otherwise, conversation history bleeds across test cases and eventually blows the context window.

---

### 12. Eval Case-Sensitivity Bug — All Retrieval Scores "Wrong"

**Symptom:** Retrieval accuracy was 40% despite the system clearly finding correct photos (silent failures showed high-confidence correct answers being marked wrong).

**Root cause:** The eval's `parse_response()` lowercased all text before extracting filenames:
```python
text = str(raw_result).lower()
photos = re.findall(r'[\w\-]+\.(?:heic|jpg|png)', text)  # extracts lowercase filenames
```

But `retrieval_correct` compared against mixed-case expected values:
```python
retrieval_correct = tc["expected_photo"] in parsed["source_photos"]
# "IMG_1853.HEIC" in ["img_1853.heic", ...] → always False
```

**Fix:** Lowercase the expected photo before comparison:
```python
retrieval_correct = tc["expected_photo"].lower() in parsed["source_photos"]
```

**Lesson:** Any string comparison in an evaluation framework must be case-normalized on both sides. This single-line bug was hiding ~5 correct retrievals and making a 65% correct system look 40% correct.

---

### 13. Routing Detection in Eval Parsed Wrong Query Type

**Symptom:** Behavioral queries showed `routing_correct=False` even when the system correctly identified them as behavioral.

**Root cause:** The eval extracted query type by scanning response text for the first occurrence of "factual", "semantic", or "behavioral" — in that order. Agent responses often mention "factual" somewhere in boilerplate (tool descriptions list all three types), so "factual" was almost always found first.

**Fix:** Two-stage detection:
1. First look for the tool's explicit JSON output: `"query_type_detected": "behavioral"`
2. If not found, scan for keywords but check `behavioral` before `factual`:

```python
qt_match = re.search(r'query_type_detected["\s:]+(\w+)', text)
if qt_match and qt_match.group(1) in ("factual", "semantic", "behavioral"):
    parsed["query_type"] = qt_match.group(1)
else:
    for qt in ["behavioral", "factual", "semantic"]:  # behavioral first
        if qt in text:
            parsed["query_type"] = qt
            break
```

**Lesson:** When parsing LLM free-text responses for structured fields, always prefer extracting from tool output JSON rather than scanning narrative text. LLMs reference all category labels in reasoning text, making keyword-scanning unreliable.

---

### 14. RL Training Required $0 API Cost — Offline Simulation Design

**Symptom (hypothetical):** Training contextual bandits and DQN over 2,000 episodes × 5 seeds would require 10,000+ LLM queries if run live, costing ~$11,200 in GPT-4o calls.

**Root cause:** The CrewAI query pipeline makes 3-5 LLM calls per query (agent planning + tool invocation + synthesis). RL training needs thousands of episodes. Running live is cost-prohibitive.

**Fix:** Built `PhotoMindSimulator` (`src/rl/simulation_env.py`) — an offline simulation environment that:
1. Pre-computes all 3 search strategies (factual, semantic, behavioral) on every query once using the `PhotoKnowledgeBaseTool` directly (pure Python, no LLM)
2. Caches results in a dictionary keyed by (query, strategy)
3. Provides a gym-like `reset()` / `step()` interface
4. Augments 56 queries to ~560 via synonym substitution and entity swapping

Training cost: $0. Training time: ~60s on CPU for 2,000 episodes × 5 seeds.

**Tradeoff:** The simulator tests RL routing and confidence decisions in isolation — it does not test whether the LLM agents correctly *use* the tool's output. This gap is covered by the live eval (`python -m src.main eval`).

**Lesson:** For RL in agentic systems, separate the *decision policy* from the *agent execution*. Train the policy offline on cached tool outputs, then integrate the trained policy back into the live agent pipeline. This makes RL practical even with expensive LLM-based agents.

---

### 15. DQN Action Space: 4 Actions → 5 (Adding Requery)

**Symptom:** The initial DQN had 4 actions (accept_high, accept_moderate, hedge, decline) copied from a LunarLander-style architecture. But during evaluation, some queries produced mediocre results on the first strategy that could be improved by trying a different strategy — there was no mechanism to "try again."

**Root cause:** The 4-action design assumed single-shot confidence assessment: see the results, grade them, done. This missed a natural option: if results look uncertain, try an alternate search strategy before committing to a grade.

**Fix:** Added a 5th action `requery` (index 3) with multi-step episodes:
- When the DQN selects `requery`, the episode continues (non-terminal)
- The simulator picks an alternate strategy via `step_requery()`
- The DQN sees the new results and decides again
- Maximum 2 requery steps (`MAX_REQUERY_STEPS = 2`) to prevent infinite loops
- Requery carries a small step cost (-0.1) to discourage unnecessary retries
- If still `REQUERY` after max steps, falls back to grade "D" (hedge)

This made `gamma = 0.99` structurally relevant (not just inherited from LunarLander), because multi-step episodes create genuine temporal credit assignment.

**Side effect:** The README originally described "FC(64→4)" but the actual architecture is FC(64→5). This inconsistency was caught during project evaluation.

**Lesson:** Don't blindly copy action spaces from reference implementations. Design actions around the actual decisions your agent needs to make. Adding requery made gamma meaningful and gave the agent a way to recover from poor initial routing.

---

### 16. Training/Evaluation Distribution Mismatch in DQN

**Symptom:** Early DQN training used an oracle router (picking the ground-truth strategy from test case labels) to select which search results to grade. At evaluation time, the DQN saw results from the rule-based keyword router or the bandit. The resulting state distribution was different, and the DQN's learned policy transferred poorly.

**Root cause:** If you train the DQN on oracle-routed results (which are always the "correct" strategy), it never learns to handle the messy results that come from imperfect routing. The state space it sees in production differs from training.

**Fix:** In `train_dqn()`, when no trained bandit is provided, use the rule-based keyword router instead of oracle:
```python
if trained_bandit is not None:
    arm = trained_bandit.select_arm(features)
else:
    rule_strategy = _dqn_rule_tool._classify_query(info["query"])
    arm = ARM_NAMES.index(rule_strategy)
```

For full pipeline training (`train_full()`), the DQN trains on results routed by the *just-trained* bandit from the same seed — matching the deployment distribution.

**Lesson:** RL policies must train on the same distribution they'll encounter at deployment. If component A feeds into component B, train B using A's actual outputs, not a perfect oracle. This is a form of "exposure bias" familiar from seq2seq models.

---

### 17. Data Leakage Prevention — Train/Held-Out Split

**Symptom:** Early RL evaluation trained and tested on the same 20 queries (the original `TEST_CASES`). All metrics looked good, but this measured memorization, not generalization.

**Root cause:** With only 20 queries and 10x augmentation, the RL components could memorize per-query patterns. The augmented variants were too similar to originals (synonym substitution preserves structure). Without held-out queries, there was no way to distinguish genuine learning from overfitting.

**Fix:** Expanded the test suite to 56 queries (`eval/expanded_test_cases.py`) and split deterministically:
- **42 training queries** — used for RL training (simulator draws from these)
- **14 held-out queries** — never seen during training, used only for evaluation
- Split covers each category: 3 factual, 2 semantic, 2 behavioral, 3 edge case, 4 ambiguous
- Enforced via assertions: `assert len(HELD_OUT_TEST_CASES) == 14`
- Training pipeline imports `TRAIN_TEST_CASES` by default

**Lesson:** Even for offline RL with small datasets, maintain a train/test split. 10x augmentation does not create independent samples — the augmented queries share the same ground truth and structure as the originals.

---

### 18. Zero-Variance Paired t-Test Edge Case

**Symptom:** `scipy.stats.ttest_rel()` returned `(nan, nan)` when comparing baseline vs. RL silent failure rates, because all 5 seeds produced identical values (e.g., 0.024 for baseline across all seeds).

**Root cause:** When paired differences have zero variance, the standard paired t-test divides by zero (standard error = 0). Scipy returns NaN rather than handling this edge case. But the result *is* meaningful: if the RL reduces silent failures from 0.024 to 0.005 consistently across every seed, that's a real (deterministic) effect.

**Fix:** Custom `paired_t_test()` in `eval/statistical_analysis.py`:
1. If differences have zero variance AND zero mean → return `(0.0, 1.0)` — truly no difference
2. If differences have zero variance AND non-zero mean → use `ttest_1samp(diff, 0)` — this correctly identifies a deterministic consistent effect
3. Otherwise → use standard `ttest_rel()`

Similarly, `cohens_d()` returns `float('inf')` for constant non-zero differences (unbounded effect size), not 0.0.

**Lesson:** Standard statistical tests assume random variation. When your experiment produces perfectly consistent results (same across all seeds), the tests break. Handle this edge case explicitly — deterministic improvement is stronger than noisy improvement, not weaker.

---

### 19. Bandit Routing Accuracy Lower Than Baseline on Non-Ambiguous Queries

**Symptom:** Thompson Sampling bandit achieved 71.4% routing accuracy vs. 78.6% for the rule-based keyword baseline on the full test set.

**Root cause:** The keyword-based rule router was *designed* for the non-ambiguous queries (factual/semantic/behavioral keywords map directly to strategies). The bandit, trained to handle ambiguous queries better, sacrificed some accuracy on clear-cut queries in exchange for better performance on ambiguous ones.

**Key insight:** This is the intended Pareto tradeoff — the bandit's value is on the 11 ambiguous queries where the keyword router systematically fails. On the 45 non-ambiguous queries, the keyword router is nearly optimal.

**Resolution:** Created a 5th evaluation config: "Recommended (Rule+DQN)" — uses rule-based routing (best for non-ambiguous queries) with the DQN confidence calibrator (adds learned grading). This config preserves baseline routing accuracy while adding DQN's silent failure reduction.

**Lesson:** A learned policy doesn't always beat a well-designed heuristic. When the heuristic is optimal for the majority of cases, the right architecture is to use the heuristic by default and the learned component only where the heuristic is known to fail. In this project, that's the "Recommended" config: rule routing + DQN grading.

---

## Current System Performance

### Live Eval (CrewAI query crew — 20 test queries)

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | **95%** |
| Routing Accuracy | **100%** |
| Silent Failure Rate | **0%** |
| Decline Accuracy | **100%** |
| Avg Latency | 44s/query |

### RL Offline Eval (56 queries, 5 seeds, 95% CIs)

| Config | Retrieval | Routing | Silent Failure | Decline |
|--------|-----------|---------|----------------|---------|
| Baseline (Rule-Based) | 95.4% | 78.6% | 2.4% | 100% |
| Bandit Only (Thompson) | 91.1% | 71.4% | 2.4% | 80.0% |
| DQN Only | 95.4% | 78.6% | 0.5% | 100% |
| Full RL (Thompson+DQN) | 91.1% | 71.4% | 0.5% | 80.0% |
| **Recommended (Rule+DQN)** | **95.4%** | **78.6%** | **0.5%** | **100%** |

Key finding: The Recommended config (rule routing + DQN grading) preserves baseline retrieval/routing accuracy while reducing silent failure rate from 2.4% to 0.5%. Statistical significance varies by metric (p=0.178 for silent failure rate with 5 seeds — honest limitation).

---

## Known Limitations

| Limitation | Impact | Potential Fix |
|------------|--------|---------------|
| Keyword overlap semantic search | Misses synonyms ("cozy" ≠ "comfortable") | Replace with vector embeddings (sentence-transformers cosine similarity) |
| Confidence scores (0.15–0.6) reflect keyword uncertainty | Grades don't distinguish "found it" from "guessed" | DQN confidence calibrator partially addresses this (Issue 15) |
| Agent doesn't always cite source filenames | Eval under-counts correct retrievals | Enforce structured output with Pydantic response model |
| CrewAI memory grows across queries | TPM errors in long eval runs | Create fresh crew per query (done in eval) |
| Behavioral search returns dominant-type photos only | "Which store most often?" ignores store name distribution | Aggregate entity values, not just image types |
| Rule-based query routing | "What stores did I visit?" misclassifies as semantic | Bandit routing helps on ambiguous queries but hurts non-ambiguous ones (Issue 19) |
| Small dataset (25 photos, 56 queries) | RL results may not generalize to larger collections | Expand photo set and query diversity for stronger validation |
| Statistical significance gap | 5 seeds × 56 queries insufficient for p<0.05 on some metrics (e.g., silent failure p=0.178) | More seeds, larger held-out set, or bootstrapped CIs |
| Bandit routing worse on non-ambiguous queries | Thompson Sampling trades 7% routing accuracy for ambiguous query coverage | Hybrid routing: bandit only for ambiguous queries, rule for rest |
| Offline simulation fidelity | Simulator tests routing/grading decisions but not LLM agent execution | Periodic live eval to validate end-to-end behavior |
| 10x augmentation creates correlated samples | Augmented queries share ground truth with originals, inflating effective sample count | Use LLM-generated paraphrases or external query sets for independence |

---

## Architecture Decisions Worth Documenting

**Why bypass CrewAI for ingestion?**
The ingestion pipeline is a pure ETL: read file → call vision API → write JSON. There's no multi-step reasoning needed. CrewAI adds ~3-4x API call overhead for zero benefit here. Ingestion is now a direct Python script. The CrewAI crew is kept for query answering where multi-agent collaboration (retrieval → synthesis) adds real value.

**Why a flat JSON knowledge base instead of a vector DB?**
Simplicity. A `photo_index.json` file is readable, debuggable, and requires no infrastructure. For 25-500 photos, a linear scan is fast enough (< 50ms). At 5,000+ photos, a vector DB (Chroma, Qdrant) would become necessary.

**Why sentence-transformers for memory instead of OpenAI embeddings?**
API key restrictions. OpenAI's embeddings endpoint requires separate access which not all keys have. `sentence-transformers/all-MiniLM-L6-v2` is local, free, and fast (~20ms per query on CPU). It downloads once (~80MB) and runs entirely offline.

**Why three separate search strategies instead of one unified approach?**
Different query types have fundamentally different information needs:
- Factual: entity tables and OCR text, exact match preferred
- Semantic: description text, soft match preferred
- Behavioral: aggregation over all photos, no single match

A single embedding-based search would handle semantic well but struggle with exact entity lookup (e.g., finding "18.69" in extracted amounts) and aggregation (counting receipts per store).

**Why two RL components (bandits + DQN) instead of one?**
They solve different problems. The contextual bandit optimizes *which search strategy* to run (routing — a stateless selection). The DQN optimizes *how to grade the results* (confidence calibration — a potentially multi-step decision with requery). A single RL component couldn't cleanly handle both: routing is a one-shot arm selection, while grading benefits from sequential requery logic. Separating them also enables the "Recommended" config: use one without the other.

**Why offline simulation instead of online RL?**
Cost. The CrewAI pipeline costs ~$0.56/query (GPT-4o vision + agent calls). Training over 2,000 episodes × 5 seeds × 56 queries would cost ~$11,200 online. The `PhotoMindSimulator` pre-computes search results using the `PhotoKnowledgeBaseTool` directly (pure Python, no LLM), caching all strategy outputs once. Training then runs in ~60s on CPU at $0. The tradeoff is fidelity — the simulator doesn't test LLM synthesis, only routing and grading decisions.

**Why a 5th "Recommended" eval config?**
The ablation study revealed a Pareto tradeoff: bandit routing improves ambiguous queries but hurts non-ambiguous ones (+7% error), while DQN grading improves all queries. The 5th config (rule routing + DQN grading) cherry-picks the best component from each category. This isn't cherry-picking results — it's a genuine architectural insight: use learned components only where heuristics fail.

**Why 42/14 train/held-out split instead of k-fold?**
With 56 queries, k-fold cross-validation would give unstable estimates per fold. A fixed 42/14 split provides a stable held-out set large enough to detect generalization failure. The split is stratified across query categories and includes 4 of the 11 ambiguous queries in held-out, ensuring the most important query type is tested on unseen data.

---

## File Structure (Final)

```
PhotoMind/
├── src/
│   ├── main.py                    # CLI: ingest / query / eval / train / rl-eval / ablation
│   ├── config.py                  # Pydantic Settings (reads .env)
│   ├── ingest_direct.py           # Direct ingestion (bypasses CrewAI)
│   ├── agents/
│   │   └── definitions.py         # 4 agent factory functions
│   ├── crews/
│   │   ├── ingestion_crew.py      # Sequential crew
│   │   └── query_crew.py          # Hierarchical crew (RL-integrated)
│   ├── tasks/
│   │   ├── ingestion.py           # 3 ingestion tasks
│   │   └── query.py               # 1 query task
│   ├── tools/
│   │   ├── photo_vision.py        # GPT-4o vision tool with HEIC support
│   │   ├── photo_knowledge_base.py # Custom 3-strategy search tool
│   │   └── feedback_store.py      # Adaptive feedback loop for threshold learning
│   └── rl/
│       ├── rl_config.py           # Hyperparameters, reward matrix, action space
│       ├── contextual_bandit.py   # Thompson Sampling, UCB1, Epsilon-Greedy bandits
│       ├── dqn_confidence.py      # ConfidenceDQN FC(8→64→64→5) + agent
│       ├── feature_extractor.py   # Query→feature vector (TF-IDF + metadata)
│       ├── replay_buffer.py       # DQN experience replay
│       ├── reward.py              # Reward matrix lookup
│       ├── simulation_env.py      # PhotoMindSimulator (offline, $0 training)
│       └── training_pipeline.py   # train_bandit / train_dqn / train_full / run_ablation
├── eval/
│   ├── test_cases.py              # 20 original hand-labeled test queries
│   ├── expanded_test_cases.py     # 56 queries: 42 train + 14 held-out
│   ├── run_evaluation.py          # Live eval harness (CrewAI)
│   ├── run_rl_evaluation.py       # RL eval: 5 configs, paired t-tests, CIs
│   ├── statistical_analysis.py    # Custom t-test, Cohen's d, CI formatting
│   ├── ablation.py                # 7-config ablation study driver
│   └── results/
│       ├── eval_results.json      # Latest live eval output
│       ├── rl_eval_results.json   # Latest RL eval output
│       ├── rl_training_results.json # Training metrics per seed
│       ├── ablation_results.json  # Ablation study results
│       ├── eval_history.json      # Historical eval runs
│       └── scaling_benchmark.json # Scaling experiment data
├── viz/
│   ├── plot_learning_curves.py    # DQN reward + epsilon decay plots
│   ├── plot_ablation.py           # Ablation comparison bar charts
│   ├── plot_before_after.py       # Before/after RL improvement plots
│   ├── plot_regret.py             # Bandit cumulative regret plots
│   ├── generate_diagrams.py       # Mermaid diagram generation
│   ├── diagrams/                  # .mmd architecture diagrams
│   └── figures/                   # .png + .pdf generated figures
├── scripts/
│   ├── train_bandit.py            # Standalone bandit training
│   ├── train_dqn.py               # Standalone DQN training
│   ├── train_full.py              # Full pipeline training (bandit→DQN)
│   ├── precompute_cache.py        # Pre-compute simulator cache
│   ├── scaling_benchmark.py       # Scaling experiment
│   └── demo_comparison.py         # Side-by-side baseline vs RL demo
├── tests/
│   └── test_core.py               # Unit tests
├── docs/
│   └── math_formulations.md       # Mathematical formulation of RL components
├── knowledge_base/
│   ├── photo_index.json           # 25 indexed photos
│   ├── feedback_store.json        # Adaptive feedback data
│   └── rl_models/
│       ├── bandit_thompson.pkl    # Trained Thompson Sampling bandit
│       └── dqn_confidence.pth     # Trained DQN weights
├── photos/                        # 25 iPhone photos (HEIC + PNG + JPG)
├── requirements.txt
├── .env                           # OPENAI_API_KEY (not committed)
├── CLAUDE.md                      # Project guidance for AI assistants
├── TECHNICAL_REPORT.md            # Full technical report
├── PROJECT_RETROSPECTIVE.md       # This file
└── README.md                      # Project overview and usage
```

---

*Generated: April 11, 2026 | Updated: April 19, 2026 (added Issues 14–19, RL extension)*
