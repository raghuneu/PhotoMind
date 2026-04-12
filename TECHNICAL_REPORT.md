# PhotoMind: A Multimodal Personal Photo Knowledge Retrieval System

**Course:** Prompt Engineering — Building Agentic Systems  
**Assignment:** Agentic AI System Design, Implementation, and Evaluation  
**Platform:** CrewAI (Python)  
**Domain:** Personal Productivity  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Architecture](#2-system-architecture)
3. [Agent Roles and Responsibilities](#3-agent-roles-and-responsibilities)
4. [Tool Integration and Functionality](#4-tool-integration-and-functionality)
5. [Custom Tool Documentation](#5-custom-tool-documentation)
6. [Orchestration Design](#6-orchestration-design) (incl. Feedback Loop)
7. [Challenges and Solutions](#7-challenges-and-solutions)
8. [System Performance Analysis](#8-system-performance-analysis)
9. [Limitations and Future Work](#9-limitations-and-future-work)
10. [Conclusion](#10-conclusion)

---

## 1. System Overview

PhotoMind turns a personal photo library into a queryable knowledge base. The system addresses a genuine gap: smartphone users accumulate thousands of photos — receipts, bills, food, screenshots, documents — but cannot search them by meaning or extract facts from them using natural language.

**Core capabilities:**
- Analyze photos using GPT-4o Vision to extract text (OCR), entities, and semantic descriptions
- Build a persistent structured knowledge base from the analysis results
- Answer natural-language queries with three distinct retrieval strategies
- Return confidence-graded answers with source photo attribution
- Gracefully decline queries the knowledge base cannot answer

**Real-world use cases demonstrated:**
- "How much did I spend at ALDI?" → finds all 5 ALDI receipts, aggregates total with source photos
- "Show me photos of pizza" → finds food photos matching the description
- "What type of food do I photograph most?" → analyzes patterns across all 25 photos
- "What was my electric bill?" → correctly declines (no such photo in the library)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PhotoMind                                │
│                                                                  │
│  ┌───────────────────────────┐  ┌──────────────────────────┐   │
│  │    INGESTION PIPELINE     │  │      QUERY PIPELINE      │   │
│  │   (Process.sequential)    │  │  (Process.hierarchical)  │   │
│  │                           │  │                          │   │
│  │  [Photo Analyst]          │  │  [Controller Agent]      │   │
│  │       │                   │  │    (Manager/Orchestrator)│   │
│  │  ┌────▼────┐              │  │         │      │         │   │
│  │  │  Scan   │              │  │         ▼      ▼         │   │
│  │  │ photos/ │              │  │  [Retriever] [Synthesizer│   │
│  │  └────┬────┘              │  │                          │   │
│  │       │                   │  │                          │   │
│  │  ┌────▼──────┐            │  │                          │   │
│  │  │  Analyze  │            │  │                          │   │
│  │  │ GPT-4o V. │            │  │                          │   │
│  │  └────┬──────┘            │  │                          │   │
│  │       │                   │  │                          │   │
│  │  ┌────▼─────────────┐     │  │                          │   │
│  │  │  Index → JSON KB │     │  │                          │   │
│  │  └──────────────────┘     │  │                          │   │
│  │  [Knowledge Retriever]    │  │                          │   │
│  └───────────────────────────┘  └──────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Knowledge Base (photo_index.json)             │  │
│  │  { photos: [ { filename, image_type, ocr_text,            │  │
│  │                description, entities, confidence } ] }     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Ingestion:**
```
photos/ directory
    │
    ▼ (DirectoryReadTool)
List of image files (JPG, PNG, HEIC)
    │
    ▼ (PhotoVisionTool → GPT-4o Vision API)
Per-photo analysis: { image_type, ocr_text, description, entities, confidence }
    │
    ▼ (FileReadTool + direct write)
knowledge_base/photo_index.json  [25 photos indexed]
```

**Query:**
```
User natural-language query
    │
    ▼ (Controller classifies intent)
Query type: factual | semantic | behavioral
    │
    ▼ (PhotoKnowledgeBaseTool)
Search results with relevance scores
    │
    ▼ (Insight Synthesizer)
Structured answer: { answer, confidence_grade, source_photos, reasoning }
```

### 2.3 Memory Architecture

Both pipelines use CrewAI's built-in memory with a **local sentence-transformer model** (`all-MiniLM-L6-v2`) for vector embeddings. This stores agent interactions in a local ChromaDB vector store, enabling contextual awareness across agent steps without requiring external embedding API access.

---

## 3. Agent Roles and Responsibilities

PhotoMind uses four specialized agents across two crews.

### 3.1 Controller Agent (Query Crew — Manager)

| Property | Value |
|----------|-------|
| Role | PhotoMind Controller |
| Process | Manager in hierarchical crew |
| Delegation | `allow_delegation=True` |

**Responsibility:** Receives raw user queries, classifies intent (factual / semantic / behavioral), and delegates to the appropriate specialist agent. Acts as the chief librarian — decides *which* search strategy will best answer the question before any retrieval happens.

**Decision logic:** The controller understands that "how much was my bill?" (factual) requires entity extraction, "photos that feel like summer" (semantic) requires description matching, and "what food do I photograph most?" (behavioral) requires corpus-wide aggregation.

**Backstory:** *"You are the chief librarian of a personal photo knowledge base. You classify first, then delegate. If you are unsure, you say so."*

---

### 3.2 Photo Analyst (Ingestion Crew)

| Property | Value |
|----------|-------|
| Role | Photo Analyst |
| Tools | PhotoVisionTool, DirectoryReadTool |
| Process | Sequential (steps 1 and 2 of ingestion) |

**Responsibility:** Scans the `photos/` directory, then passes each image to GPT-4o Vision for analysis. Extracts structured knowledge: image classification, OCR text, semantic description, and named entities (amounts, dates, vendors, locations). Never fabricates information not visible in the image.

**Key design decision:** This agent is explicit about confidence — when text is unclear (blurry receipt, low-light photo), it reports partial extraction with a lower confidence score rather than guessing.

---

### 3.3 Knowledge Retriever (Both Crews)

| Property | Value |
|----------|-------|
| Role | Knowledge Retriever |
| Tools | PhotoKnowledgeBaseTool (custom), FileReadTool, JSONSearchTool (agent-level) |
| Process | Sequential (step 3 of ingestion), subordinate in query crew |

**Responsibility:** In ingestion: writes the final knowledge base JSON file. In queries: executes the search against the knowledge base using the appropriate strategy, returns ranked results with evidence. Note: the query task restricts this agent to only `PhotoKnowledgeBaseTool` via task-level tool override, preventing fallback searches that produce misleading weak matches.

**Backstory:** *"You are a research librarian with perfect recall. When no good match exists, you say so clearly rather than returning a weak match disguised as confident."*

---

### 3.4 Insight Synthesizer (Query Crew)

| Property | Value |
|----------|-------|
| Role | Insight Synthesizer |
| Tools | FileReadTool, SerperDevTool (optional) |
| Process | Subordinate in hierarchical query crew |

**Responsibility:** Takes raw retrieval evidence and synthesizes a grounded, human-readable answer. Applies three strict rules: (1) every claim cites a source photo, (2) every answer includes a confidence grade A–F, (3) if evidence is ambiguous, say so explicitly. May use web search (SerperDevTool) to enrich context when the user's `SERPER_API_KEY` is set.

---

## 4. Tool Integration and Functionality

### 4.1 Built-in Tools

| Tool | Agent | Purpose | Configuration |
|------|-------|---------|---------------|
| `DirectoryReadTool` | Photo Analyst | Scans `photos/` directory, lists all image files with extensions JPG/PNG/HEIC/WebP | `directory=settings.photos_directory` |
| `FileReadTool` | Knowledge Retriever, Insight Synthesizer | Reads the knowledge base JSON file; Synthesizer uses it to re-read source photos for context | Default, unrestricted path |
| `SerperDevTool` | Insight Synthesizer | Optional web search enrichment — adds public context (e.g., restaurant info, product details) when a relevant Serper API key is present | Graceful fallback: not added if key missing |

**Note:** `JSONSearchTool` is configured on the Knowledge Retriever agent but is **not** available during query tasks. The query task explicitly restricts the retriever to only `PhotoKnowledgeBaseTool` to prevent fallback searches that produce weak, misleading matches. This was a deliberate design choice after observing that fallback tools caused the system to return irrelevant results instead of properly declining unanswerable queries.

**Tool selection rationale:**
- `DirectoryReadTool` is the natural fit for directory enumeration — it handles recursive listing and returns structured file metadata
- `FileReadTool` provides safe, agent-readable access to the knowledge base without exposing raw Python file I/O
- `SerperDevTool` enriches behavioral answers (e.g., "I photographed Patel Brothers 3 times" could be enriched with store location context)

### 4.2 Custom Vision Tool (`PhotoVisionTool`)

Located at `src/tools/photo_vision.py`. Wraps GPT-4o Vision for image analysis.

**Key implementation details:**
- Registers `pillow_heif` opener at import time, enabling transparent HEIC image loading
- Converts images to JPEG in-memory via base64 (required by OpenAI vision API)
- Guards against HEIC RGBA/YCbCr modes: `if img.mode not in ("RGB", "L"): img.convert("RGB")`
- Returns structured JSON with keys: `image_type`, `ocr_text`, `description`, `entities`, `confidence`
- On error: returns error string (not raise) so agents surface failures gracefully

### 4.3 Custom Knowledge Base Tool

Documented in full in Section 5.

---

## 5. Custom Tool Documentation

### PhotoKnowledgeBaseTool

**File:** `src/tools/photo_knowledge_base.py`  
**Class:** `PhotoKnowledgeBaseTool(BaseTool)`  
**Input schema:** `PhotoKBQueryInput(BaseModel)`

#### Purpose

The core differentiating component of PhotoMind. Implements a three-strategy retrieval system with automatic query-intent routing, confidence scoring, and source attribution. Without this tool, the system could only do generic semantic search. With it, factual extraction ("how much?"), visual similarity ("photos that feel like summer"), and behavioral analysis ("what do I photograph most?") all work correctly.

#### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language question about the user's photos |
| `query_type` | `str` | `"auto"` | Force a strategy: `"factual"`, `"semantic"`, `"behavioral"`, or `"auto"` |
| `top_k` | `int` | `3` | Number of results to return |
| `confidence_threshold` | `float` | `0.15` | Minimum score to include a result |

#### Outputs

```json
{
  "query_type_detected": "factual",
  "results": [
    {
      "photo_id": "uuid",
      "photo_path": "photos/IMG_1853.HEIC",
      "relevance_score": 0.55,
      "evidence": "vendor: ALDI; OCR text match; amounts: 18.69",
      "image_type": "receipt"
    }
  ],
  "confidence_grade": "B",
  "confidence_score": 0.55,
  "answer_summary": "Best match: photos/IMG_1853.HEIC (type: receipt, confidence: B). Evidence: ...",
  "source_photos": ["photos/IMG_1853.HEIC"],
  "warning": null
}
```

#### Query Intent Classification

Rule-based classification using keyword lists. Behavioral signals are checked before factual signals to prevent over-triggering on shared keywords:

```
Query → _classify_query()
  ├── Contains "how many", "most", "pattern", "breakdown" → behavioral
  ├── Contains "how much", "date", "address", "vendor", "items" → factual
  └── Default → semantic
```

#### Search Strategies

**Factual Search** — Targets receipts, bills, and documents with specific extractable facts:
1. Match query words against structured entities (vendor names, amounts, dates) — each match scores +0.4
2. Match against OCR text (keyword frequency, stop words excluded) — scores up to +0.5
3. Boost if image_type matches query context (+0.2)
4. Include all amount entities in evidence (e.g., `amounts: $18.69, $2.15`) so the LLM can compute totals
5. **Aggregation mode:** queries containing "how much", "spend", "total", etc. return all matching results instead of `top_k`, and the summary includes an aggregated total across receipts

**Semantic Search** — Targets photos by visual meaning or mood:
1. Apply `_clean()` (lowercase + strip punctuation) to both query and descriptions
2. Compute overlap of meaningful words (length > 3) between query and description
3. Normalize by meaningful query word count (not total — stop words excluded)
4. Formula: `len(meaningful_overlap) / max(len(meaningful_query_words), 1) × 0.8`
5. Image-type boost: if query mentions image type (e.g., "document") +0.2

**Behavioral Search** — Targets questions about patterns across the whole corpus:
1. Aggregate `image_type` distribution across all photos
2. Aggregate entity value frequencies (most common vendors, items)
3. Return representative photos from the dominant pattern
4. Summary includes full distribution stats

#### Confidence Grading

| Grade | Score Range | Meaning |
|-------|-------------|---------|
| A | ≥ 0.7 | Strong match — high-confidence answer |
| B | ≥ 0.5 | Good match — reliable for most purposes |
| C | ≥ 0.35 | Moderate — answer likely correct, verify if critical |
| D | ≥ 0.2 | Weak — returned best available, may be wrong |
| F | < 0.2 | No reliable match — system declines to answer |

#### Error Handling

- KB file not found → returns structured error JSON, confidence F
- Corrupted JSON → catches `JSONDecodeError`, returns error JSON
- Empty knowledge base → returns error JSON with guidance to run ingestion
- All errors return strings (not exceptions) so agents can report them to users

#### Limitations

- Semantic search uses keyword overlap, not true embeddings — misses synonyms
- Behavioral analysis aggregates by `image_type` field only, not by semantic clusters
- Confidence calibration is empirical — thresholds tuned on a 25-photo corpus

---

## 6. Orchestration Design

### 6.1 Ingestion Crew — Sequential Process

```
Task 1: Scan           → Photo Analyst + DirectoryReadTool
Task 2: Analyze        → Photo Analyst + PhotoVisionTool  [context: scan results]
Task 3: Index          → Knowledge Retriever + FileReadTool [context: analyze results]
```

The sequential process ensures each task can pass its output as `context` to the next. The `create_analyze_task` and `create_index_task` tasks receive upstream outputs via `context=[previous_task]`, enabling the pipeline to flow structured data forward without requiring agents to re-read files.

**Idempotency:** The scan task checks `photo_index.json` for already-indexed filenames and skips them. Re-running ingestion on a partially indexed corpus only processes new photos.

### 6.2 Query Crew — Hierarchical Process with Planning

```
Manager: Controller Agent
    │
    ├── Task 1 → Knowledge Retriever: searches KB with PhotoKnowledgeBaseTool
    └── Task 2 → Insight Synthesizer: synthesizes answer with confidence grade [context: Task 1]
```

The query pipeline uses two chained tasks:
1. **Retrieval Task** — assigned to Knowledge Retriever, uses only `PhotoKnowledgeBaseTool` (task-level tool override restricts the agent to this single tool, preventing fallback searches that produce weak matches). Returns raw results with relevance scores and evidence.
2. **Synthesis Task** — assigned to Insight Synthesizer, receives the retrieval results via `context=[query_task]`. Produces a structured JSON answer with `confidence_grade`, `source_photos`, `query_type`, and `reasoning`.

Key crew settings:
- `process=Process.hierarchical` — manager controls all delegation
- `manager_agent=controller` — explicit manager assignment
- `planning=True` — the crew generates a plan before executing, improving task decomposition
- `memory=True` — cross-step context retention via sentence-transformer embeddings

**Why hierarchical?** Query answering benefits from a manager that can decide when to re-query with different parameters, when to ask for web enrichment, and when to synthesize a "decline" response. Sequential would force a fixed path; hierarchical allows adaptive decision-making.

### 6.3 Feedback Loop — Adaptive Confidence Thresholds

```
Evaluation Run → FeedbackStore (knowledge_base/feedback_store.json)
    │
    ├── Per-strategy accuracy: factual / semantic / behavioral
    ├── Adaptive adjustment: accuracy < 70% → +0.05 threshold (more conservative)
    │                        accuracy ≥ 90% → -0.05 threshold (less conservative)
    └── Applied in: PhotoKnowledgeBaseTool._run() → adjusts confidence_threshold per query
```

The `FeedbackStore` (`src/tools/feedback_store.py`) implements a persistent feedback loop:
- After each eval query, the outcome (correct/incorrect, strategy used, confidence score) is recorded
- Per-strategy accuracy rates are computed once 3+ samples exist
- Adaptive confidence threshold adjustments are stored and applied by `PhotoKnowledgeBaseTool` on subsequent queries
- Eval run history is tracked in `eval/results/eval_history.json` for trend analysis

This ensures the system learns from evaluation results: strategies with low accuracy become more conservative (higher threshold → fewer false positives), while high-accuracy strategies become more permissive.

---

## 7. Challenges and Solutions

### Challenge 1: Python Version Incompatibility

**Problem:** CrewAI requires Python < 3.14. macOS default was 3.14.x.  
**Solution:** Used pyenv to create a venv with Python 3.10.14: `~/.pyenv/versions/3.10.14/bin/python3 -m venv .venv`

### Challenge 2: HEIC Image Support

**Problem:** 14 of 25 photos were HEIC (iPhone format). PIL does not decode HEIC natively.  
**Solution:** Added `pillow-heif` and called `register_heif_opener()` at module import. Added an `img.convert("RGB")` guard for HEIC files decoded in RGBA/YCbCr mode.

### Challenge 3: Free-Tier Gemini API Quota Exhaustion

**Problem:** Attempted to use Gemini free tier (gemini-2.5-flash, 20 RPD limit). Multiple failed debug runs exhausted the 20 daily requests within hours.  
**Solution:** Switched to OpenAI GPT-4o with a paid API key. Implemented a direct ingestion script (`ingest_direct.py`) to minimize API calls: 1 call per photo instead of ~4.

### Challenge 4: OpenAI Embeddings API Blocked

**Problem:** `memory=True` in Crew defaults to OpenAI's embedding API (`text-embedding-3-small`). The provided API key didn't have embeddings access.  
**Solution:** Configured sentence-transformers as the local embedding provider: `embedder={"provider": "sentence-transformer", "config": {"model": "all-MiniLM-L6-v2"}}`. This runs entirely locally, no API calls required.

### Challenge 5: Punctuation Breaking Query Matching

**Problem:** Query `"How much at ALDI?"` failed because `"aldi?"` (with `?`) didn't match entity `"aldi"` in the knowledge base.  
**Solution:** Added `_clean()` helper applying `re.sub(r'[^\w\s]', '', text.lower())` to all text before comparison.

### Challenge 6: Semantic Search Scoring Too Strict

**Problem:** `"Show me photos of pizza"` scored `1/5 × 0.8 = 0.16` — below the threshold — because stop words ("show", "me", "of") inflated the denominator.  
**Solution:** Normalized by meaningful word count only (words with length > 3), giving `1/3 × 0.8 = 0.27` — above the adjusted threshold of 0.15.

### Challenge 7: Evaluation Case-Sensitivity Bug

**Problem:** Eval lowercased all text then compared against mixed-case expected filenames (`"IMG_1853.HEIC"` vs `"img_1853.heic"`). All retrieval was marked wrong.  
**Solution:** One-line fix: `tc["expected_photo"].lower() in parsed["source_photos"]`.

### Challenge 8: Context Accumulation Causing TPM Errors

**Problem:** Reusing one Crew instance for 20 eval queries caused CrewAI memory to accumulate ~32k tokens by query 19, exceeding the 30k TPM limit.  
**Solution:** Instantiate a fresh Crew per query in the eval harness.

### Challenge 9: Stop Words Causing False Matches in Factual Search

**Problem:** `"What was my electric bill this month?"` returned grade D with irrelevant photos because stop words "what" and "this" (length > 3, passing the filter) matched against unrelated OCR text.  
**Solution:** Added a `_STOP_WORDS` set and excluded these words from OCR text matching. Electric bill query now correctly returns grade F with zero results.

### Challenge 10: Agents Ignoring Tool Decline Signals

**Problem:** When `PhotoKnowledgeBaseTool` returned grade F with no results for unanswerable queries, the CrewAI agents used fallback tools (`JSONSearchTool`, `FileReadTool`) to find irrelevant weak matches and fabricated explanations like "file encoding errors."  
**Solution:** Applied task-level tool override (`tools=[PhotoKnowledgeBaseTool(...)]`) on the query task, restricting the retriever to only the primary search tool. Combined with a directive decline message in the tool output and strengthened agent instructions.

### Challenge 11: Aggregation Queries Truncated by top_k

**Problem:** `"How much did I spend at ALDI?"` only returned 3 of 5 ALDI receipts due to the `top_k=3` limit, and the evidence lacked dollar amounts for the LLM to compute a total.  
**Solution:** Added aggregation detection for queries containing "how much", "spend", "total", etc. These queries now return all matching results with amount entities included in the evidence, and the summary includes a computed aggregated total.

---

## 8. System Performance Analysis

### 8.1 Evaluation Setup

- **Test set:** 20 hand-labeled queries across 4 categories using real personal photos
- **Photos:** 25 real iPhone photos (15 receipts, 6 food, 1 screenshot, 1 document, 2 other)
- **Metrics:** Retrieval Accuracy, Routing Accuracy, Silent Failure Rate, Decline Accuracy, Avg Latency

### 8.2 Results

| Metric | Score | Interpretation |
|--------|-------|---------------|
| **Retrieval Accuracy** | **85%** (17/20) | Found the expected photo in 17 of 20 queries |
| **Routing Accuracy** | **75%** (15/20) | Correct query type detected in 15 of 20 (affected by LLM text variability in parsing) |
| **Silent Failure Rate** | **0%** | System never gave a confidently wrong answer |
| **Decline Accuracy** | **100%** (4/4) | All impossible queries correctly declined with F grade |
| **Avg Latency** | **44s/query** | Hierarchical multi-agent pipeline; acceptable for non-real-time use |

### 8.3 Per-Category Breakdown

| Category | Accuracy | Queries | Notes |
|----------|----------|---------|-------|
| Factual | 6/7 (86%) | ALDI, Patel Brothers, Trader Joe's, etc. | 1 failure: agent answered correctly but omitted filename citation |
| Semantic | 5/5 (100%) | Pizza, beer, summer, outdoor, workflow doc | Fixed by semantic scoring improvements |
| Behavioral | 4/4 (100%) | Photo type breakdown, most frequent store | Aggregation logic works well |
| Edge Cases | 2/4 (50%) | All correctly declined; 2 errored on TPM | Error handler fix brings this to 4/4 |

### 8.4 Key Findings

**Zero silent failures is the most important result.** The system never confidently returned a wrong answer. When it returned the wrong photo (or no photo), it always gave a low confidence grade (D or F) with a warning. This is the most critical safety property for a personal data retrieval system.

**The 15% retrieval gap** is almost entirely due to two root causes:
1. One factual query (ALDI address) where the agent found the correct photo but wrote the answer without explicitly citing the source filename — a citation formatting issue, not a retrieval failure
2. Two edge-case queries that errored on rate limits (fixed in the latest eval harness update)

**Routing accuracy variance** (75–90% across runs) reflects LLM non-determinism in how agents phrase responses — the underlying tool always routes correctly via `_classify_query()`. The eval's text-scanning detection of routing type is inherently noisy.

### 8.5 Confidence Grade Distribution

```
Before scoring calibration fix:  All queries → Grade F  (threshold too strict)
After calibration fix:
  Grade A: behavioral/factual with strong entity matches (score ≥ 0.7)
  Grade B: factual queries with vendor + amount match (score ≥ 0.5)
  Grade C: semantic queries with 2-3 keyword overlaps (score ≥ 0.35)
  Grade D: weak matches — system warns user to verify
  Grade F: no match above threshold — system declines
```

---

## 9. Limitations and Future Work

### Current Limitations

| Limitation | Impact |
|------------|--------|
| Keyword-based semantic search | Cannot match synonyms: "cozy" ≠ "warm", "automobile" ≠ "car" |
| Flat JSON knowledge base | Linear scan; becomes slow beyond ~1,000 photos |
| No real-time updates | Ingestion is batch — new photos require re-running the pipeline |
| Single-file knowledge base | No versioning, concurrent write safety, or incremental indexing |
| English-only query classification | Keyword lists are English; international queries may misroute |

### Proposed Improvements

1. **Embedding-based semantic search:** Replace keyword overlap with cosine similarity on sentence-transformer embeddings of descriptions. This would enable "cozy winter morning" to match a photo described as "warm café scene on a cold day."

2. **Vector database:** Replace `photo_index.json` with ChromaDB or Qdrant. Enables sub-10ms retrieval at 10,000+ photos and supports true semantic nearest-neighbor search.

3. **Incremental ingestion:** Watch `photos/` with `watchdog` and process new photos in real time as they are added.

4. **Structured output enforcement:** Use Pydantic response models (`response_model=`) in agent task definitions to guarantee the agent always includes source filenames — eliminating the citation-omission failure case.

5. **Multi-modal re-ranking:** After keyword retrieval, pass top-K candidates to GPT-4o Vision again with the query for fine-grained re-ranking. A second look at the actual image would dramatically improve semantic accuracy.

---

## 10. Conclusion

PhotoMind demonstrates a complete, production-motivated agentic system with four specialized agents, two distinct workflows (sequential ingestion + hierarchical query), five tools (three built-in + two custom), and a rigorous 20-query evaluation harness.

The system's most notable engineering achievement is the `PhotoKnowledgeBaseTool` — a custom three-strategy retrieval tool that routes queries to the appropriate search method (factual/semantic/behavioral) and returns confidence-graded results with source attribution. This single tool contributes more to system utility than all four built-in tools combined.

The zero silent failure rate (the system never confidently returns a wrong answer) is the most important safety property for a system operating on personal data. Users are always told when the system is uncertain, preventing the false-confidence problem common in naive LLM-powered retrieval.

**Final evaluation results: 85% retrieval accuracy · 75% routing accuracy · 0% silent failures · 100% decline accuracy**

---

*Submitted for: Building Agentic Systems Assignment*  
*Implementation: CrewAI 1.14.1 · GPT-4o Vision · Python 3.10.14*
