"""
PhotoKnowledgeBaseTool — Custom tool for PhotoMind.

Searches the personal photo knowledge base with query-intent routing,
confidence scoring (A-F), and source photo attribution. Implements four
distinct search strategies: factual, semantic, behavioral, and embedding.

Inputs:
  - query (str): Natural language question about the user's photos
  - query_type (str): "factual", "semantic", "behavioral", "embedding", or "auto"
  - top_k (int): Number of results to return (default 3)
  - confidence_threshold (float): Minimum score to include (default 0.3)

Outputs:
  - JSON with: query_type_detected, results[], confidence_grade (A-F),
    confidence_score, answer_summary, source_photos[], warning (if low confidence)

Search Strategy Overview:
  - factual: Entity + OCR keyword matching for precise fact extraction
  - semantic: Weighted keyword overlap with descriptions (fast, deterministic)
  - embedding: Dense vector cosine similarity via all-MiniLM-L6-v2 (captures
    paraphrases, zero API cost, runs locally)
  - behavioral: Frequency aggregation across the indexed corpus

The RL contextual bandit selects among these four strategies at query time,
and the DQN confidence calibrator grades the result quality.
"""

from __future__ import annotations

from typing import Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import json
import os
import re

from src.tools.feedback_store import FeedbackStore


def _clean(text: str) -> str:
    """Lowercase and strip punctuation so 'aldi?' matches 'aldi'."""
    return re.sub(r'[^\w\s]', '', text.lower())


# Stop words that should not contribute to relevance scoring
_STOP_WORDS = frozenset({
    "what", "when", "where", "which", "this", "that", "with", "from",
    "have", "been", "were", "your", "does", "will", "would", "could",
    "should", "about", "their", "there", "these", "those", "much",
    "many", "some", "than", "them", "then", "they", "into", "each",
})


class PhotoKBQueryInput(BaseModel):
    """Input schema for PhotoMind Knowledge Base queries."""

    query: str = Field(
        ...,
        description="Natural language query about the user's photos"
    )
    query_type: str = Field(
        default="auto",
        description=(
            "Query intent type: 'factual' (amounts, dates, specific facts), "
            "'semantic' (visual similarity, descriptions, mood), "
            "'behavioral' (patterns, frequency, preferences), "
            "or 'auto' (let the tool classify)"
        )
    )
    top_k: int = Field(
        default=3,
        description="Number of top results to return"
    )
    confidence_threshold: float = Field(
        default=0.25,
        description="Minimum confidence score (0-1) to include a result"
    )


class PhotoKnowledgeBaseTool(BaseTool):
    """
    Searches the PhotoMind personal photo knowledge base with query-intent
    routing, multi-strategy retrieval, confidence scoring, and source attribution.
    """

    name: str = "photo_knowledge_base"
    description: str = (
        "Searches the PhotoMind personal photo knowledge base. Supports four "
        "query types: 'factual' for extracting specific facts from photos "
        "(bill amounts, dates, vendor names), 'semantic' for finding photos by "
        "keyword overlap with descriptions, 'embedding' for dense vector "
        "similarity search (paraphrases, conceptual matches), and 'behavioral' "
        "for analyzing patterns across photos (most photographed food, spending "
        "trends). Returns results with confidence scores and source photo "
        "attribution. ALWAYS use this tool when answering questions about the "
        "user's photos."
    )
    args_schema: Type[BaseModel] = PhotoKBQueryInput
    knowledge_base_path: str = "./knowledge_base/photo_index.json"
    # Optional repository — when set, live queries use the repo instead of
    # reading the JSON file on every call.  Left as None for backward compat
    # (RL simulation passes a raw kb dict directly to search methods).
    repository: Any = Field(default=None, exclude=True)

    def _run(
        self,
        query: str,
        query_type: str = "auto",
        top_k: int = 3,
        confidence_threshold: float = 0.25,
    ) -> str:
        """Execute a query against the photo knowledge base."""
        try:
            # Load data: prefer repository, fall back to JSON file
            if self.repository is not None:
                kb = {"photos": self.repository.all_photos()}
            elif not os.path.exists(self.knowledge_base_path):
                return json.dumps({
                    "error": f"Knowledge base not found at {self.knowledge_base_path}. "
                             "Run the ingestion pipeline first.",
                    "confidence_grade": "F",
                    "confidence_score": 0.0
                })
            else:
                with open(self.knowledge_base_path, "r") as f:
                    kb = json.load(f)

            if not kb.get("photos"):
                return json.dumps({
                    "error": "Knowledge base is empty. Run ingestion first.",
                    "confidence_grade": "F",
                    "confidence_score": 0.0,
                })

            # Auto-classify query type: try RL bandit first, fall back to rules
            routing_source = "rule_based"
            if query_type == "auto":
                rl_type = self._rl_classify_query(query)
                if rl_type is not None:
                    # Guard: reject bandit's "behavioral" pick when the query
                    # lacks any behavioral-intent signal — short noun queries
                    # like "pizza" get misrouted and produce generic results.
                    if rl_type == "behavioral" and not self._has_behavioral_intent(query):
                        query_type = self._classify_query(query)
                        routing_source = "rule_based"
                    # Symmetric guard: reject bandit's non-behavioral pick
                    # when the query has strong behavioral intent.  Queries
                    # like "how many receipts" and "which store do I shop at
                    # most" were being routed to factual by the bandit.
                    elif rl_type != "behavioral" and self._has_behavioral_intent(query):
                        rule_type = self._classify_query(query)
                        if rule_type == "behavioral":
                            query_type = "behavioral"
                            routing_source = "rule_based"
                        else:
                            query_type = rl_type
                            routing_source = "rl_bandit"
                    else:
                        query_type = rl_type
                        routing_source = "rl_bandit"
                else:
                    query_type = self._classify_query(query)
                    routing_source = "rule_based"

            # Apply feedback-based confidence threshold adjustment
            strategy_accuracy = None
            try:
                feedback = FeedbackStore()
                adjustment = feedback.get_confidence_adjustment(query_type)
                confidence_threshold += adjustment
                strategy_accuracy = feedback.get_strategy_accuracy(query_type)
            except Exception:
                pass  # Feedback store is optional; gracefully degrade

            # Detect aggregation queries that need all matches
            q_clean = _clean(query)
            aggregation_keywords = ["total", "how much", "spend", "spent", "all", "sum"]
            is_aggregation = any(kw in q_clean for kw in aggregation_keywords)

            # Route to appropriate search strategy.
            # When a vector-capable repository is available, embedding and
            # semantic queries upgrade to hybrid (RRF) search automatically.
            use_hybrid = (
                self.repository is not None
                and hasattr(self.repository, "embedding_search")
                and query_type in ("embedding", "semantic")
            )

            if use_hybrid:
                results = self._hybrid_search(query, kb, top_k)
            elif query_type == "factual":
                results = self._factual_search(query, kb, top_k)
            elif query_type == "behavioral":
                results = self._behavioral_search(query, kb, top_k)
            elif query_type == "embedding":
                results = self._embedding_search(query, kb, top_k)
            else:
                results = self._semantic_search(query, kb, top_k)

            # Filter by confidence threshold
            filtered = [r for r in results if r["relevance_score"] >= confidence_threshold]

            # For aggregation queries, keep all results; otherwise limit to top_k
            output_results = filtered if is_aggregation else filtered[:top_k]

            # Calculate confidence grade
            # Minimum-score floor: if the best result is below this threshold
            # the query is effectively unanswerable — treat as no results.
            _RELEVANCE_FLOOR = 0.25
            if not output_results or output_results[0]["relevance_score"] < _RELEVANCE_FLOOR:
                grade, score = "F", 0.0
                output_results = []  # discard noise-level matches
                warning = (
                    "NO MATCHING PHOTOS FOUND. The knowledge base does not contain "
                    "information related to this query. You MUST decline to answer "
                    "and report confidence_grade F with an empty source_photos list."
                )
            else:
                score = output_results[0]["relevance_score"]
                dqn_grade = self._rl_confidence_grade(output_results, query_type, query)
                grade = dqn_grade if dqn_grade is not None else self._score_to_grade(score)

                # P4: Multi-evidence factual calibration — when multiple
                # strong matches corroborate a factual query, promote the
                # grade by one step (B→A, C→B). Under-calibrated factual
                # answers with 2+ receipts at >=0.5 were frequently judged
                # correct but returned as "B" in sweep evaluations.
                if query_type == "factual":
                    strong_matches = sum(
                        1 for r in output_results if r["relevance_score"] >= 0.5
                    )
                    if strong_matches >= 2:
                        _promote = {"B": "A", "C": "B", "D": "C"}
                        grade = _promote.get(grade, grade)

                warning = None
                if grade in ("D", "F"):
                    warning = "Low confidence results. Please verify against source photos."

            response = {
                "query_type_detected": query_type,
                "routing_rationale": (
                    f"Strategy '{query_type}' selected via {routing_source}. "
                    + (f"RL bandit overrode rule-based default."
                       if routing_source == "rl_bandit"
                       else "Rule-based classifier matched query keywords.")
                ),
                "results": output_results,
                "confidence_grade": grade,
                "confidence_score": round(score, 3),
                "answer_summary": self._generate_summary(query, output_results),
                "source_photos": [r["photo_path"] for r in output_results],
                "warning": warning,
                "strategy_accuracy": strategy_accuracy,
            }
            return json.dumps(response, indent=2)

        except json.JSONDecodeError:
            return json.dumps({
                "error": "Knowledge base file is corrupted or contains invalid JSON.",
                "confidence_grade": "F",
                "confidence_score": 0.0,
            })
        except Exception as e:
            return json.dumps({
                "error": f"Unexpected error during search: {str(e)}",
                "confidence_grade": "F",
                "confidence_score": 0.0,
            })

    # ── RL-Powered Query Routing ────────────────────────────────────────

    def _get_rl_components(self):
        """Lazy-load and cache RL models and feature extractor at instance level."""
        if not hasattr(self, "_rl_cache"):
            self._rl_cache = {}
        if not self._rl_cache:
            try:
                from src.rl.contextual_bandit import load_trained_bandit
                from src.rl.dqn_confidence import load_trained_dqn
                from src.rl.feature_extractor import QueryFeatureExtractor
                from src.rl.rl_config import BANDIT_MODEL_PATH, DQN_MODEL_PATH

                self._rl_cache["bandit"] = load_trained_bandit(BANDIT_MODEL_PATH)
                self._rl_cache["dqn"] = load_trained_dqn(DQN_MODEL_PATH)
                self._rl_cache["extractor"] = QueryFeatureExtractor(self.knowledge_base_path)
            except Exception:
                self._rl_cache = {}  # reset so we retry next time
        return self._rl_cache

    def _rl_classify_query(self, query: str) -> str | None:
        """Use trained contextual bandit for query routing. Returns None on failure."""
        try:
            from src.rl.rl_config import ARM_NAMES

            cache = self._get_rl_components()
            bandit = cache.get("bandit")
            extractor = cache.get("extractor")
            if bandit is None or extractor is None:
                return None
            features = extractor.extract(query)
            arm = bandit.select_arm(features)
            return ARM_NAMES[arm]
        except Exception:
            return None

    def _rl_confidence_grade(self, results: list, strategy: str, query: str) -> str | None:
        """Use trained DQN for confidence grading. Returns None on failure.

        Supports the multi-step *requery* action: when the DQN selects requery,
        we run an alternate search strategy and let the DQN re-evaluate.
        """
        try:
            from src.rl.dqn_confidence import ConfidenceState, action_to_grade
            from src.rl.rl_config import ARM_NAMES, MAX_REQUERY_STEPS
            import random as _rand

            cache = self._get_rl_components()
            agent = cache.get("dqn")
            extractor = cache.get("extractor")
            if agent is None or extractor is None:
                return None
            features = extractor.extract(query)
            strategy_idx = ARM_NAMES.index(strategy) if strategy in ARM_NAMES else 1
            state = ConfidenceState.from_retrieval(results, strategy_idx, features)
            action = agent.select_action(state)
            score = results[0]["relevance_score"] if results else 0.0
            grade = action_to_grade(action, score)

            # Handle requery: try alternate strategy up to MAX_REQUERY_STEPS
            requery_count = 0
            best_score = score  # track best score across requery attempts
            current_strategy_idx = strategy_idx
            while grade == "REQUERY" and requery_count < MAX_REQUERY_STEPS:
                alt_arms = [a for a in range(len(ARM_NAMES)) if a != current_strategy_idx]
                new_arm = _rand.choice(alt_arms)
                new_strategy = ARM_NAMES[new_arm]
                with open(self.knowledge_base_path) as f:
                    kb = json.load(f)
                search_fn = {
                    "factual": self._factual_search,
                    "semantic": self._semantic_search,
                    "behavioral": self._behavioral_search,
                    "embedding": self._embedding_search,
                }.get(new_strategy, self._semantic_search)
                results = search_fn(query, kb, top_k=5)
                current_strategy_idx = new_arm
                state = ConfidenceState.from_retrieval(results, new_arm, features)
                action = agent.select_action(state)
                score = results[0]["relevance_score"] if results else 0.0
                best_score = max(best_score, score)
                grade = action_to_grade(action, score)
                requery_count += 1

            if grade == "REQUERY":
                # Exhausted requery budget — fall back to rule-based grading
                # using the best score observed across all attempts.
                grade = self._score_to_grade(best_score)

            return grade
        except Exception:
            return None

    # ── Query Intent Classification ──────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """Rule-based query intent classification."""
        q = _clean(query)

        # ── Behavioral-priority patterns ──
        # These multi-word patterns indicate aggregation / counting / comparison
        # intent even when factual keywords like "receipt" co-occur.
        # Checked BEFORE the factual keyword list to avoid misrouting
        # "how many receipts" → factual (wrong) instead of behavioral (correct).
        behavioral_priority_patterns = [
            "how many",           # "how many receipts do I have"
            "how often",          # "how often do I shop"
            "which store",        # "which store do I shop at most"
            "which.*most",        # "which type do I have the most"
            "which.*highest",     # "which receipt had the highest total"
            "do i.*more",         # "do I shop more at ALDI or TJ"
            "do i ever",          # "do I ever pay with cash" — habit/frequency
            "most expensive",     # "what is the most expensive purchase"
            "most frequent",      # "what is the most frequent"
            "shop at most",       # "where do I shop at most"
            "total across",       # "what is the total across all receipts"
            "across all",         # "across all receipts"
            "on average",         # "what do I spend on average"
            "percentage of",      # "what percentage of my photos are..."
            "what cuisines",      # "what cuisines do I eat"
            "what types of",      # "what types of stores do I have"
            "what type of food",  # "what type of food do I eat most"
            "what kind of food",
            "what cuisine",
            "how frequently",
            "how regularly",
            "tend to",            # "do I tend to shop at..."
            "more.*or.*less",     # "am I eating more or less ..."
            "more.*or.*takeout", # "home-cooked meals or takeout"
        ]
        import re as _re
        for pat in behavioral_priority_patterns:
            if _re.search(pat, q):
                return "behavioral"

        factual_keywords = [
            "how much", "what amount", "total", "price", "cost", "date",
            "when", "bill", "receipt", "invoice", "payment", "account number",
            "phone number", "address", "due", "vendor", "company", "items",
            "buy", "bought", "purchase", "purchased", "paid for",
            "bill for", "receipt for", "spend at", "spent at", "order from",
            "balance", "owe", "paid",
        ]
        behavioral_keywords = [
            "most", "often", "frequently", "pattern", "trend", "favorite",
            "usually", "habit", "how many times", "how many", "what kind of",
            "what type of", "distribution", "breakdown",
            "compare", "versus", "vs", "prefer", "average",
        ]
        semantic_keywords = [
            "show me", "look like", "feel like", "similar to", "remind",
            "photos of", "pictures of", "find photos", "find pictures",
            "scenic", "outdoor", "beautiful", "mood", "find a",
        ]

        # Check factual first — precise entity/amount queries take priority
        # over aggregation keywords that may co-occur (e.g. "how much total")
        if any(kw in q for kw in factual_keywords):
            return "factual"
        elif any(kw in q for kw in behavioral_keywords):
            return "behavioral"
        elif any(kw in q for kw in semantic_keywords):
            return "semantic"
        else:
            return "semantic"

    def _has_behavioral_intent(self, query: str) -> bool:
        """Check if query contains behavioral-intent keywords."""
        import re as _re
        q = _clean(query)
        behavioral_signals = [
            "most", "often", "frequently", "pattern", "trend", "favorite",
            "usually", "habit", "how many times", "how many", "what kind of",
            "what type of", "distribution", "breakdown",
            "compare", "versus", "vs", "prefer", "average",
        ]
        # Also check multi-word patterns that are strong behavioral signals
        behavioral_patterns = [
            "how many", "how often", "which store", "which.*most",
            "which.*highest", "do i.*more", "do i ever",
            "most expensive", "most frequent",
            "total across", "across all", "on average",
            "percentage of", "what cuisines", "what types of",
            "more.*or.*less",             "more.*or.*takeout",
        ]
        if any(kw in q for kw in behavioral_signals):
            return True
        return any(_re.search(pat, q) for pat in behavioral_patterns)

    # ── Search Strategies ────────────────────────────────────────────────

    def _factual_search(self, query: str, kb: dict, top_k: int) -> list:
        """Search extracted entities and OCR text for factual answers."""
        results = []
        q = _clean(query)
        words = q.split()
        # Meaningful tokens: long enough to be distinctive and not a stop word.
        # Using the same filter for entity-value AND OCR branches keeps scoring
        # consistent and prevents fillers ("how", "did") from matching abstract
        # entities like topic="How to build AI agents" on a spending query.
        meaningful_words = [w for w in words if len(w) > 3 and w not in _STOP_WORDS]

        # Identify "distinctive" query terms — proper nouns / brand names that
        # are NOT generic category words.  If the query contains a distinctive
        # term and a result matches only generic terms but NOT the distinctive
        # one, the result gets penalised.  This prevents "Netflix subscription
        # receipt" from returning random receipts at grade A.
        _GENERIC_CATEGORY_WORDS = {
            "receipt", "receipts", "bill", "bills", "photo", "photos",
            "picture", "pictures", "document", "documents", "invoice",
            "invoices", "subscription", "order", "purchase", "payment",
            "find", "show", "total",
        }
        distinctive_words = [w for w in meaningful_words
                             if w.lower() not in _GENERIC_CATEGORY_WORDS]

        for photo in kb["photos"]:
            score = 0.0
            evidence_parts = []

            # Match against structured entities (token-boundary, meaningful words only)
            for entity in photo.get("entities", []):
                val = _clean(entity.get("value", ""))
                etype = entity.get("type", "").lower()
                val_tokens = set(val.split())
                if any(w in val_tokens for w in meaningful_words):
                    score += 0.4
                    evidence_parts.append(f"{etype}: {entity['value']}")
                    # Boost if entity type also matches query context
                    # (only when the value already matched — avoids inflating
                    # scores on generic type words like "amount" or "date")
                    if etype in q.split():
                        score += 0.1

            # Match against OCR text (uses the same meaningful-word filter)
            ocr_text = _clean(photo.get("ocr_text", ""))
            if ocr_text:
                matching = sum(1 for w in meaningful_words if w in ocr_text)
                word_score = min(matching * 0.15, 0.5)
                score += word_score
                if word_score > 0:
                    evidence_parts.append("OCR text match")

            # Image type bonus (guard against empty string — "" in q is always True)
            img_type = photo.get("image_type", "").lower()
            if img_type and img_type in q:
                score += 0.2

            score = min(score, 1.0)

            # Distinctive-term penalty: if the query has brand-name / proper
            # noun terms and NONE of them appear in this photo's entities or
            # OCR, the match is likely coincidental on generic words.
            if distinctive_words and score > 0:
                photo_blob = (
                    " ".join(_clean(e.get("value", "")) for e in photo.get("entities", []))
                    + " " + _clean(photo.get("ocr_text", ""))
                ).lower()
                distinctive_hits = sum(1 for w in distinctive_words if w in photo_blob)
                if distinctive_hits == 0:
                    score = min(score, 0.20)

            if score > 0:
                # Deduplicate evidence
                evidence_parts = list(dict.fromkeys(evidence_parts))

                # Include all amount entities so the LLM can compute totals
                amounts = [
                    e["value"] for e in photo.get("entities", [])
                    if e.get("type", "").lower() == "amount"
                ]
                if amounts:
                    evidence_parts.append(f"amounts: {', '.join(amounts)}")

                results.append({
                    "photo_id": photo["id"],
                    "photo_path": photo["file_path"],
                    "relevance_score": round(score, 3),
                    "evidence": "; ".join(evidence_parts) or "Partial match",
                    "image_type": photo.get("image_type", "unknown"),
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _semantic_search(self, query: str, kb: dict, top_k: int) -> list:
        """Search descriptions and captions for semantic matches.

        Design Trade-off — Keyword Overlap vs. Dense Embeddings
        --------------------------------------------------------
        This method uses *weighted keyword overlap* (Jaccard-like) rather
        than dense-vector cosine similarity.  This is a conscious design
        decision with clear trade-offs:

        **Pros of keyword overlap (chosen approach):**
        - Zero external dependency: no embedding model needed at query
          time, enabling fully offline operation.
        - Transparent scoring: every point of relevance_score is
          attributable to a specific word match, aiding debuggability.
        - Deterministic: identical queries always produce identical results
          (important for RL training reproducibility).

        **Cons (and mitigation):**
        - Cannot capture paraphrase equivalence ("expensive meal" ≠
          "pricey dinner").  *Mitigated* by the RL bandit's ability to
          re-route a semantic query to factual search when keyword overlap
          fails — the requery action in the DQN provides a second chance.
        - Score ceiling is lower (~0.8 max) than cosine similarity (~1.0),
          meaning the grade boundaries in ``_score_to_grade`` are
          calibrated specifically for this distributional range.

        A future improvement would be to add a dense-embedding fallback
        (e.g., ``sentence-transformers/all-MiniLM-L6-v2``) as a fourth
        bandit arm, letting the RL agent learn when keyword overlap
        suffices vs. when embeddings are needed.
        """
        results = []
        query_words = set(_clean(query).split())
        # Normalize by meaningful query words only (ignore stop words like "me", "of")
        meaningful_query_words = {w for w in query_words if len(w) > 3}

        # Negative-entity guard: if the query names a specific entity (capitalized
        # word in original query, len>3, not a stop word), we want to suppress
        # receipts whose vendor/entity list clearly does NOT include it.
        import re as _re_ent
        raw_tokens = _re_ent.findall(r"[A-Za-z][A-Za-z0-9'\-]+", query)
        cap_entities = {
            t.lower() for t in raw_tokens
            if len(t) > 3 and t[0].isupper() and t.lower() not in {
                "what", "when", "where", "which", "show", "find", "have",
                "many", "much", "does", "this", "that", "with", "from",
            }
        }

        for photo in kb["photos"]:
            # Clean text to strip punctuation so "joe's" matches "joes"
            description = _clean(photo.get("description", ""))
            caption = _clean(photo.get("caption", ""))
            combined = f"{description} {caption}"

            if not combined.strip():
                continue

            combined_words = set(combined.split())
            overlap = query_words & combined_words
            # Ignore very short words (a, the, is, etc.)
            meaningful_overlap = {w for w in overlap if len(w) > 3}
            # Normalize by meaningful query word count with a floor of 2.
            # A 1-word query like "pizza" that exactly matches a description
            # now scores 0.8/2 = 0.4 (grade C) — comfortably above the
            # relevance floor (0.25) and feedback-raised thresholds, while
            # still well below A (>=0.7) so a single keyword cannot bluff
            # high confidence.
            score = len(meaningful_overlap) / max(len(meaningful_query_words), 2) * 0.8

            # Image type relevance boost
            img_type = photo.get("image_type", "").lower()
            if img_type and img_type in query.lower():
                score += 0.2

            # Direct-substring boost: when a meaningful query word appears
            # verbatim in the description text (not just token overlap), give
            # +0.1 to make exact-noun queries like "pizza" / "beer" robust
            # against feedback threshold drift.
            if meaningful_overlap and any(w in description for w in meaningful_overlap):
                score += 0.1

            # P5: Single-noun / entity expansion — when token overlap alone
            # yields nothing, also try matching against the photo's entities
            # and image_type. Example: "beer" matches a photo with
            # entity value "beer" or image_type "beverage_receipt".
            photo_entity_values = {
                _clean(e.get("value", ""))
                for e in photo.get("entities", [])
                if e.get("value")
            }
            photo_entity_tokens = set()
            for ev in photo_entity_values:
                photo_entity_tokens.update(ev.split())

            if score < 0.35 and meaningful_query_words:
                entity_hits = meaningful_query_words & photo_entity_tokens
                type_hit = img_type and any(
                    w in img_type for w in meaningful_query_words
                )
                if entity_hits:
                    score = max(score, 0.4 + 0.1 * min(len(entity_hits), 3))
                elif type_hit:
                    score = max(score, 0.35)

            # P3: Negative-entity hallucination guard — if the query names a
            # specific capitalized entity (e.g. "Netflix", "ALDI") and this
            # photo is a receipt whose vendor/entity list does NOT mention
            # that entity at all, strongly suppress it. Prevents grocery
            # receipts from surfacing for "Netflix bill" queries.
            if cap_entities and photo.get("image_type", "").lower() == "receipt":
                photo_entity_blob = " ".join(photo_entity_values) + " " + description
                if not any(ent in photo_entity_blob for ent in cap_entities):
                    score *= 0.3

            score = min(score, 1.0)
            if score > 0:
                results.append({
                    "photo_id": photo["id"],
                    "photo_path": photo["file_path"],
                    "relevance_score": round(score, 3),
                    "evidence": f"Description: {photo.get('description', 'N/A')[:200]}",
                    "image_type": photo.get("image_type", "unknown"),
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results  # caller applies top_k after relevance-floor filtering

    def _behavioral_search(self, query: str, kb: dict, top_k: int) -> list:
        """Analyze patterns across photos for behavioral queries."""
        q = _clean(query)
        query_words = set(q.split())
        total_photos = len(kb["photos"])

        # Aggregate by image type
        type_counts = {}
        for photo in kb["photos"]:
            t = photo.get("image_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        # Aggregate entities grouped by entity type
        entity_type_counts = {}
        for photo in kb["photos"]:
            for entity in photo.get("entities", []):
                etype = entity.get("type", "").lower()
                val = entity.get("value", "")
                if etype not in entity_type_counts:
                    entity_type_counts[etype] = {}
                entity_type_counts[etype][val] = entity_type_counts[etype].get(val, 0) + 1

        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        # Determine query-relevant target type or entity type
        target_type = None
        for img_type in type_counts:
            if img_type in q:
                target_type = img_type
                break

        # Map query words to entity types
        entity_type_map = {
            "vendor": ["store", "shop", "vendor", "company", "where"],
            "food_item": ["food", "eat", "cook", "grocery", "item", "meal"],
            "location": ["location", "place", "city"],
            "amount": ["spend", "cost", "expensive", "cheap", "price"],
        }
        target_entity_type = None
        for etype, keywords in entity_type_map.items():
            if any(kw in q for kw in keywords):
                target_entity_type = etype
                break

        # Build summary (entity bullets) and capture top-entity set for evidence grounding
        summary_parts = [f"Photo type distribution: {dict(sorted_types[:5])}"]
        top_entities: list[str] = []  # Preserves rank order; index 0 = #1 frequent
        if target_entity_type and target_entity_type in entity_type_counts:
            sorted_entities = sorted(
                entity_type_counts[target_entity_type].items(),
                key=lambda x: x[1], reverse=True
            )
            summary_parts.append(
                f"Most frequent {target_entity_type}s: {dict(sorted_entities[:5])}"
            )
            top_entities = [v for v, _ in sorted_entities[:5]]
        else:
            all_entities = {}
            for etype_counts in entity_type_counts.values():
                for val, count in etype_counts.items():
                    all_entities[val] = all_entities.get(val, 0) + count
            sorted_all = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)
            if sorted_all:
                summary_parts.append(f"Most frequent entities: {dict(sorted_all[:5])}")
                top_entities = [v for v, _ in sorted_all[:5]]

        # ── Evidence-grounded photo selection ──
        # Primary pool: photos that contain at least one of the top aggregated entities
        # (these photos substantiate the answer bullets). Pool is independent of
        # image_type so that OCR-derived entities correctly attribute to receipts.
        top_entity_set = {v.lower() for v in top_entities if v}

        def _match_top_entity(photo: dict) -> tuple[int, str | None]:
            """Return (match_count, best_matched_entity_by_rank) for a photo.

            A photo's stored entity only counts as evidence when its value is
            also substring-grounded in the photo's OCR or description. This
            prevents silent mismatches where the KB lists an entity that does
            not textually appear in what the user can read/see on the photo.
            """
            source_blob = (
                (photo.get("ocr_text") or "") + " "
                + (photo.get("description") or "") + " "
                + (photo.get("caption") or "")
            ).lower()
            grounded_vals: set[str] = set()
            for e in photo.get("entities", []):
                val = (e.get("value") or "").strip()
                if not val:
                    continue
                val_clean = _clean(val)
                if not val_clean:
                    continue
                # Whole-phrase match, else token-majority match (>= half of tokens >2 chars)
                if val_clean in source_blob:
                    grounded_vals.add(val_clean)
                    continue
                toks = [t for t in val_clean.split() if len(t) > 2]
                if toks:
                    hits = sum(1 for t in toks if t in source_blob)
                    if hits >= max(1, len(toks) // 2 + len(toks) % 2):
                        grounded_vals.add(val_clean)
            count = sum(1 for v in grounded_vals if v in top_entity_set)
            best = None
            for ranked in top_entities:  # Iterate in rank order
                if _clean(ranked) in grounded_vals:
                    best = ranked
                    break
            return count, best

        primary_pool: list[tuple[dict, int, str | None]] = []
        if top_entity_set:
            for p in kb["photos"]:
                cnt, best = _match_top_entity(p)
                if cnt > 0:
                    primary_pool.append((p, cnt, best))

        # Secondary pool: photos with the user's literal image_type frame, or the
        # dominant type when the user didn't name one.
        if target_type:
            secondary_type = target_type
        else:
            secondary_type = sorted_types[0][0] if sorted_types else "unknown"
        secondary_photos = [p for p in kb["photos"]
                            if p.get("image_type") == secondary_type]

        # Pick working pool: primary when non-empty, else secondary.
        used_primary = bool(primary_pool)
        if used_primary:
            working: list[tuple[dict, int, str | None]] = primary_pool
            max_match = max(cnt for _, cnt, _ in working) or 1
        else:
            working = [(p, 0, None) for p in secondary_photos]
            max_match = 1

        # Provenance: where did the answer actually come from?
        if used_primary:
            source_types: dict[str, int] = {}
            for p, _, _ in working:
                t = p.get("image_type", "unknown")
                source_types[t] = source_types.get(t, 0) + 1
            dominant_source = max(source_types, key=source_types.get) if source_types else "photo"
            provenance = f"Based on {len(working)} {dominant_source} photos"
            summary_parts.insert(0, provenance)

        results = []
        top1 = top_entities[0] if top_entities else None
        for photo, match_cnt, best_entity in working:
            score = 0.1  # Baseline

            # Direct evidential support — the core signal
            if used_primary:
                score += 0.4 * (match_cnt / max_match)
                if top1 and best_entity == top1:
                    score += 0.2

            # Image-type affinity (user's literal frame)
            if photo.get("image_type") == target_type:
                score += 0.2

            # Fallback signals when there's no primary pool
            if not used_primary:
                photo_type = _clean(photo.get("image_type", ""))
                if photo_type in query_words:
                    score += 0.3
                matched_type = target_type or (sorted_types[0][0] if sorted_types else None)
                if matched_type and total_photos > 0:
                    score += (type_counts.get(matched_type, 0) / total_photos) * 0.4

            # Cap: primary-match path can reach 1.0, but fallback path
            # (no direct entity evidence) is capped at 0.55 to prevent
            # grade-A inflation on purely type-frequency guesses.
            cap = 1.0 if used_primary else 0.55
            score = min(score, cap)

            results.append({
                "photo_id": photo["id"],
                "photo_path": photo["file_path"],
                "relevance_score": round(score, 3),
                "evidence": "; ".join(summary_parts),
                "image_type": photo.get("image_type", "unknown"),
                "matched_entity": best_entity,
            })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def _embedding_search(self, query: str, kb: dict, top_k: int) -> list:
        """Dense vector similarity search using sentence-transformers.

        Uses all-MiniLM-L6-v2 (384-dim, local, zero API cost) to encode both
        the query and each photo's combined text (description + OCR + entities).
        Cosine similarity is computed via dot product on L2-normalized vectors.

        When a repository with embedding_search is available (Qdrant), the
        query vector is sent to the repo for server-side ANN search.  Otherwise
        falls back to the local npz brute-force index.
        """
        try:
            from src.tools.embedding_index import _get_model
        except ImportError:
            return self._semantic_search(query, kb, top_k)

        try:
            model = _get_model()
            query_vec = model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")[0]

            # Prefer repository vector search (Qdrant ANN)
            if self.repository is not None and hasattr(self.repository, "embedding_search"):
                raw_results = self.repository.embedding_search(query_vec, top_k)
                results = []
                for r in raw_results:
                    photo = r.get("photo", {})
                    results.append({
                        "photo_id": r["photo_id"],
                        "photo_path": photo.get("file_path", ""),
                        "relevance_score": round(r["score"], 3),
                        "evidence": (
                            f"Embedding similarity: {r['score']:.3f} | "
                            f"{photo.get('description', '')[:150]}"
                        ),
                        "image_type": photo.get("image_type", "unknown"),
                    })
                return results

            # Fallback: local EmbeddingIndex (npz cache)
            from src.tools.embedding_index import EmbeddingIndex
            idx = EmbeddingIndex(kb_path=self.knowledge_base_path)
            idx.load(photos=kb.get("photos", []))
            raw_results = idx.search(query, top_k=top_k)
        except Exception:
            return self._semantic_search(query, kb, top_k)

        # Build a photo lookup for enriching results
        photo_map = {p["id"]: p for p in kb.get("photos", [])}

        results = []
        for r in raw_results:
            photo = photo_map.get(r["photo_id"], {})
            results.append({
                "photo_id": r["photo_id"],
                "photo_path": photo.get("file_path", ""),
                "relevance_score": round(r["score"], 3),
                "evidence": f"Embedding similarity: {r['score']:.3f} | {r['text_preview'][:150]}",
                "image_type": photo.get("image_type", "unknown"),
            })
        return results

    # ── Hybrid Search (RRF) ────────────────────────────────────────────

    def _hybrid_search(
        self, query: str, kb: dict, top_k: int
    ) -> list[dict]:
        """Reciprocal Rank Fusion of embedding + factual results.

        score(doc) = Σ  1 / (k + rank_i)   with k = 60

        Merges the ranked lists from embedding_search (semantic depth) and
        factual_search (entity precision), producing a single blended list.
        Only used when a repository with vector search is available.

        **Relevance gate**: If the best raw embedding similarity is below
        _HYBRID_MIN_SIMILARITY, the query is effectively unrelated to the
        knowledge base and RRF would only amplify noise.  In that case we
        return an empty list so the caller grades as F.
        """
        _HYBRID_MIN_SIMILARITY = 0.35  # raw cosine-sim floor
        RRF_K = 60

        # Gather ranked lists from two complementary strategies
        embedding_results = self._embedding_search(query, kb, top_k=top_k * 2)
        factual_results = self._factual_search(query, kb, top_k=top_k * 2)

        # ── Relevance gate ──
        # Check whether the best embedding result has meaningful similarity.
        # Embedding scores are raw cosine similarities (0-1).  When the best
        # score is below the threshold, the query has no semantic match in the
        # KB — RRF normalization would map noise-level results to high scores.
        best_emb_sim = 0.0
        if embedding_results:
            best_emb_sim = max(r["relevance_score"] for r in embedding_results)
        best_fact_score = 0.0
        if factual_results:
            best_fact_score = max(r["relevance_score"] for r in factual_results)

        # If neither strategy found a meaningful match, bail out early.
        if best_emb_sim < _HYBRID_MIN_SIMILARITY and best_fact_score < 0.25:
            return []

        # Build RRF score map keyed by photo_id
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}
        # Track raw embedding similarity per photo for score capping
        raw_emb_sim: dict[str, float] = {}

        for rank, r in enumerate(embedding_results):
            pid = r["photo_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            raw_emb_sim[pid] = r["relevance_score"]
            if pid not in result_map:
                result_map[pid] = r

        for rank, r in enumerate(factual_results):
            pid = r["photo_id"]
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            if pid not in result_map:
                result_map[pid] = r

        # Sort by fused score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        if not sorted_ids:
            return []

        # ── Score normalization with similarity anchoring ──
        # Instead of blindly normalizing to 0.2-1.0, anchor the final score
        # to the actual relevance signal.  The raw embedding similarity is
        # the most honest relevance indicator; RRF rank is just an ordering
        # heuristic.  Final score = weighted blend of RRF rank-score and
        # the raw embedding similarity, ensuring low-similarity results
        # never inflate to grade A.
        max_rrf = rrf_scores[sorted_ids[0]]
        min_rrf = rrf_scores[sorted_ids[-1]] if len(sorted_ids) > 1 else 0.0
        rrf_range = max_rrf - min_rrf

        results = []
        for pid in sorted_ids[:top_k]:
            r = result_map[pid].copy()
            if rrf_range > 0:
                rrf_norm = (rrf_scores[pid] - min_rrf) / rrf_range
            else:
                rrf_norm = 0.5

            emb_sim = raw_emb_sim.get(pid, 0.0)
            # Blend: 40% RRF rank position + 60% raw embedding similarity.
            # This prevents RRF normalization from inflating irrelevant results.
            blended = 0.4 * rrf_norm + 0.6 * emb_sim
            # Hard cap: when embedding similarity is below the threshold,
            # the result is semantically unrelated regardless of keyword
            # overlap.  Cap score to prevent grade inflation.
            if emb_sim < _HYBRID_MIN_SIMILARITY:
                blended = min(blended, 0.20)

            r["relevance_score"] = round(blended, 4)
            r["evidence"] = f"[hybrid/RRF emb={emb_sim:.3f}] {r.get('evidence', '')}"
            results.append(r)
        return results

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade.
        Calibrated for keyword-based retrieval where top scores are ~0.4–0.8.
        """
        if score >= 0.7:
            return "A"
        if score >= 0.5:
            return "B"
        if score >= 0.35:
            return "C"
        if score >= 0.2:
            return "D"
        return "F"

    def _generate_summary(self, query: str, results: list) -> str:
        """Generate a human-readable summary of results."""
        if not results:
            return "No matching photos found in the knowledge base."
        top = results[0]
        summary = (
            f"Best match: {top['photo_path']} "
            f"(type: {top['image_type']}, "
            f"confidence: {self._score_to_grade(top['relevance_score'])}). "
            f"Evidence: {top['evidence']}"
        )

        # For multiple results with amounts, compute and append a total
        all_amounts = []
        for r in results:
            for part in r.get("evidence", "").split("; "):
                if part.startswith("amounts: "):
                    for val in part[len("amounts: "):].split(", "):
                        cleaned = re.sub(r'[^\d.\-]', '', val)
                        try:
                            all_amounts.append(float(cleaned))
                        except ValueError:
                            pass
        if len(all_amounts) > 1:
            summary += f" | Aggregated total across {len(results)} receipts: ${sum(all_amounts):.2f}"

        return summary
