"""
PhotoKnowledgeBaseTool — Custom tool for PhotoMind.

Searches the personal photo knowledge base with query-intent routing,
confidence scoring (A-F), and source photo attribution. Implements three
distinct search strategies: factual, semantic, and behavioral.

Inputs:
  - query (str): Natural language question about the user's photos
  - query_type (str): "factual", "semantic", "behavioral", or "auto"
  - top_k (int): Number of results to return (default 3)
  - confidence_threshold (float): Minimum score to include (default 0.3)

Outputs:
  - JSON with: query_type_detected, results[], confidence_grade (A-F),
    confidence_score, answer_summary, source_photos[], warning (if low confidence)

Limitations:
  - Semantic search uses keyword overlap, not true embeddings (JSONSearchTool
    handles embedding-based search separately)
  - Behavioral analysis is limited to frequency counts across the indexed corpus
  - Confidence calibration depends on the eval harness — thresholds may need tuning
"""

from typing import Type
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
        default=0.15,
        description="Minimum confidence score (0-1) to include a result"
    )


class PhotoKnowledgeBaseTool(BaseTool):
    """
    Searches the PhotoMind personal photo knowledge base with query-intent
    routing, multi-strategy retrieval, confidence scoring, and source attribution.
    """

    name: str = "photo_knowledge_base"
    description: str = (
        "Searches the PhotoMind personal photo knowledge base. Supports three "
        "query types: 'factual' for extracting specific facts from photos "
        "(bill amounts, dates, vendor names), 'semantic' for finding photos by "
        "meaning or visual description, and 'behavioral' for analyzing patterns "
        "across photos (most photographed food, spending trends). "
        "Returns results with confidence scores and source photo attribution. "
        "ALWAYS use this tool when answering questions about the user's photos."
    )
    args_schema: Type[BaseModel] = PhotoKBQueryInput
    knowledge_base_path: str = "./knowledge_base/photo_index.json"

    def _run(
        self,
        query: str,
        query_type: str = "auto",
        top_k: int = 3,
        confidence_threshold: float = 0.15,
    ) -> str:
        """Execute a query against the photo knowledge base."""
        try:
            # Validate knowledge base exists
            if not os.path.exists(self.knowledge_base_path):
                return json.dumps({
                    "error": f"Knowledge base not found at {self.knowledge_base_path}. "
                             "Run the ingestion pipeline first.",
                    "confidence_grade": "F",
                    "confidence_score": 0.0
                })

            with open(self.knowledge_base_path, "r") as f:
                kb = json.load(f)

            if not kb.get("photos"):
                return json.dumps({
                    "error": "Knowledge base is empty. Run ingestion first.",
                    "confidence_grade": "F",
                    "confidence_score": 0.0,
                })

            # Auto-classify query type if needed
            if query_type == "auto":
                query_type = self._classify_query(query)

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

            # Route to appropriate search strategy
            if query_type == "factual":
                results = self._factual_search(query, kb, top_k)
            elif query_type == "behavioral":
                results = self._behavioral_search(query, kb, top_k)
            else:
                results = self._semantic_search(query, kb, top_k)

            # Filter by confidence threshold
            filtered = [r for r in results if r["relevance_score"] >= confidence_threshold]

            # For aggregation queries, keep all results; otherwise limit to top_k
            output_results = filtered if is_aggregation else filtered[:top_k]

            # Calculate confidence grade
            if not output_results:
                grade, score = "F", 0.0
                warning = (
                    "NO MATCHING PHOTOS FOUND. The knowledge base does not contain "
                    "information related to this query. You MUST decline to answer "
                    "and report confidence_grade F with an empty source_photos list."
                )
            else:
                score = output_results[0]["relevance_score"]
                grade = self._score_to_grade(score)
                warning = None
                if grade in ("D", "F"):
                    warning = "Low confidence results. Please verify against source photos."

            response = {
                "query_type_detected": query_type,
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

    # ── Query Intent Classification ──────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        """Rule-based query intent classification."""
        q = _clean(query)

        behavioral_keywords = [
            "most", "often", "frequently", "pattern", "trend", "favorite",
            "usually", "habit", "how many times", "how many", "what kind of",
            "what type of", "distribution", "breakdown", "more", "less",
            "compare", "versus", "vs", "prefer", "average",
        ]
        factual_keywords = [
            "how much", "what amount", "total", "price", "cost", "date",
            "when", "bill", "receipt", "invoice", "payment", "account number",
            "phone number", "address", "due", "vendor", "company", "items",
            "number", "balance", "owe", "paid",
        ]
        semantic_keywords = [
            "show me", "look like", "feel like", "similar to", "remind",
            "photos of", "pictures of", "find photos", "find pictures",
            "scenic", "outdoor", "beautiful", "mood", "find a",
        ]

        # Check behavioral first — "how many receipts" should be behavioral, not factual
        if any(kw in q for kw in behavioral_keywords):
            return "behavioral"
        elif any(kw in q for kw in factual_keywords):
            return "factual"
        elif any(kw in q for kw in semantic_keywords):
            return "semantic"
        else:
            return "semantic"

    # ── Search Strategies ────────────────────────────────────────────────

    def _factual_search(self, query: str, kb: dict, top_k: int) -> list:
        """Search extracted entities and OCR text for factual answers."""
        results = []
        q = _clean(query)
        words = q.split()

        for photo in kb["photos"]:
            score = 0.0
            evidence_parts = []

            # Match against structured entities
            for entity in photo.get("entities", []):
                val = _clean(entity.get("value", ""))
                etype = entity.get("type", "").lower()
                if any(word in val for word in words if len(word) > 2):
                    score += 0.4
                    evidence_parts.append(f"{etype}: {entity['value']}")
                # Boost if entity type matches query context
                if etype in q.split():
                    score += 0.1
                    evidence_parts.append(f"{etype}: {entity['value']}")

            # Match against OCR text (exclude stop words)
            ocr_text = _clean(photo.get("ocr_text", ""))
            if ocr_text:
                matching = sum(
                    1 for w in words
                    if w in ocr_text and len(w) > 3 and w not in _STOP_WORDS
                )
                word_score = min(matching * 0.15, 0.5)
                score += word_score
                if word_score > 0:
                    evidence_parts.append("OCR text match")

            # Image type bonus
            if photo.get("image_type", "").lower() in q:
                score += 0.2

            score = min(score, 1.0)
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
        """Search descriptions and captions for semantic matches."""
        results = []
        query_words = set(_clean(query).split())
        # Normalize by meaningful query words only (ignore stop words like "me", "of")
        meaningful_query_words = {w for w in query_words if len(w) > 3}

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
            # Normalize by meaningful query word count, not total query length
            score = len(meaningful_overlap) / max(len(meaningful_query_words), 1) * 0.8

            # Image type relevance boost
            img_type = photo.get("image_type", "").lower()
            if img_type and img_type in query.lower():
                score += 0.2

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
        return results[:top_k]

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

        # Build summary
        summary_parts = [f"Photo type distribution: {dict(sorted_types[:5])}"]
        if target_entity_type and target_entity_type in entity_type_counts:
            sorted_entities = sorted(
                entity_type_counts[target_entity_type].items(),
                key=lambda x: x[1], reverse=True
            )
            summary_parts.append(
                f"Most frequent {target_entity_type}s: {dict(sorted_entities[:5])}"
            )
        else:
            all_entities = {}
            for etype_counts in entity_type_counts.values():
                for val, count in etype_counts.items():
                    all_entities[val] = all_entities.get(val, 0) + count
            sorted_all = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)
            if sorted_all:
                summary_parts.append(f"Most frequent entities: {dict(sorted_all[:5])}")

        # Select relevant photos and compute query-aware scores
        if target_type:
            relevant_photos = [p for p in kb["photos"]
                               if p.get("image_type") == target_type]
        else:
            # Fall back to dominant type
            dominant_type = sorted_types[0][0] if sorted_types else "unknown"
            relevant_photos = [p for p in kb["photos"]
                               if p.get("image_type") == dominant_type]

        results = []
        for photo in relevant_photos:
            score = 0.0

            # Query keyword overlap with photo type
            photo_type = _clean(photo.get("image_type", ""))
            if photo_type in query_words:
                score += 0.3

            # Frequency ratio: how dominant is this pattern?
            matched_type = target_type or (sorted_types[0][0] if sorted_types else None)
            if matched_type and total_photos > 0:
                freq_ratio = type_counts.get(matched_type, 0) / total_photos
                score += freq_ratio * 0.4

            # Entity match to query keywords
            for entity in photo.get("entities", []):
                val = _clean(entity.get("value", ""))
                if any(w in val for w in query_words if len(w) > 2):
                    score += 0.2
                    break

            # Behavioral aggregation always has some baseline value
            score = max(score, 0.1)
            score = min(score, 1.0)

            results.append({
                "photo_id": photo["id"],
                "photo_path": photo["file_path"],
                "relevance_score": round(score, 3),
                "evidence": "; ".join(summary_parts),
                "image_type": photo.get("image_type", "unknown"),
            })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    # ── Confidence Scoring ───────────────────────────────────────────────

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
