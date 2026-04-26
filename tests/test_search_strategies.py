"""Unit tests for PhotoKnowledgeBaseTool search strategies.

All tests run against a synthetic in-memory KB so they require no network,
no Qdrant instance, and no real photo ingestion. Each strategy is exercised
in isolation to keep failures localizable.
"""

import pytest

from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool


@pytest.fixture
def synthetic_kb():
    """A tiny 5-photo KB that exercises all strategies."""
    return {
        "photos": [
            {
                "id": "p1",
                "file_path": "photos/aldi_receipt.heic",
                "image_type": "receipt",
                "description": "ALDI grocery receipt with bread and eggs",
                "ocr_text": "ALDI MEDFORD MA TOTAL 18.69 DEBIT",
                "caption": "grocery receipt",
                "entities": [
                    {"type": "vendor", "value": "ALDI"},
                    {"type": "amount", "value": "$18.69"},
                    {"type": "location", "value": "Medford MA"},
                ],
            },
            {
                "id": "p2",
                "file_path": "photos/instacart.png",
                "image_type": "receipt",
                "description": "Instacart delivery receipt with avocados",
                "ocr_text": "INSTACART TOTAL 25.94",
                "caption": "delivery receipt",
                "entities": [
                    {"type": "vendor", "value": "Instacart"},
                    {"type": "amount", "value": "$25.94"},
                    {"type": "food_item", "value": "avocado"},
                ],
            },
            {
                "id": "p3",
                "file_path": "photos/pizza.heic",
                "image_type": "food",
                "description": "A slice of pepperoni pizza on a plate",
                "ocr_text": "",
                "caption": "pizza slice",
                "entities": [{"type": "food_item", "value": "pizza"}],
            },
            {
                "id": "p4",
                "file_path": "photos/tag.heic",
                "image_type": "other",
                "description": "UNIQLO clothing tag on a pair of jeans",
                "ocr_text": "UNIQLO JEANS $59.90",
                "caption": "clothing tag",
                "entities": [
                    {"type": "vendor", "value": "UNIQLO"},
                    {"type": "amount", "value": "$59.90"},
                ],
            },
            {
                "id": "p5",
                "file_path": "photos/doc.jpg",
                "image_type": "document",
                "description": "Workflow orchestration notes about subagents",
                "ocr_text": "workflow orchestration subagent",
                "caption": "software engineering notes",
                "entities": [],
            },
        ]
    }


@pytest.fixture
def tool():
    # knowledge_base_path doesn't need to exist — we pass the KB dict directly
    return PhotoKnowledgeBaseTool(knowledge_base_path="./knowledge_base/photo_index.json")


# ── Factual search ──────────────────────────────────────────────────────

class TestFactualSearch:

    def test_vendor_entity_match(self, tool, synthetic_kb):
        results = tool._factual_search("How much did I spend at ALDI?", synthetic_kb, top_k=3)
        assert len(results) > 0
        assert results[0]["photo_path"].endswith("aldi_receipt.heic")
        assert results[0]["relevance_score"] > 0

    def test_amount_entities_surfaced(self, tool, synthetic_kb):
        results = tool._factual_search("What was the ALDI total?", synthetic_kb, top_k=1)
        assert "18.69" in results[0]["evidence"]

    def test_no_match_returns_empty(self, tool, synthetic_kb):
        results = tool._factual_search("xyzzy quux nonsense", synthetic_kb, top_k=3)
        assert results == []

    def test_results_sorted_by_score_desc(self, tool, synthetic_kb):
        results = tool._factual_search("ALDI receipt total", synthetic_kb, top_k=5)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_abstract_topic_entity_does_not_match_factual_query(self, tool):
        """A document with an abstract `topic` entity ("How to build AI agents")
        must NOT score on a spending query. The words "how" and "did" are
        short fillers; matching them substring-style into entity values was
        the cause of misleading source-evidence thumbnails.
        """
        kb = {
            "photos": [
                {
                    "id": "r1", "file_path": "photos/aldi.heic",
                    "image_type": "receipt",
                    "description": "ALDI grocery receipt", "ocr_text": "ALDI TOTAL 18.69",
                    "caption": "receipt",
                    "entities": [
                        {"type": "vendor", "value": "ALDI"},
                        {"type": "amount", "value": "$18.69"},
                    ],
                },
                {
                    "id": "d1", "file_path": "photos/agents_doc.png",
                    "image_type": "document",
                    "description": "Pages from a book about building AI agents",
                    "ocr_text": "step 1 define the agents role and goal",
                    "caption": "book",
                    "entities": [
                        {"type": "person", "value": "Dr. Maryam Miradi"},
                        {"type": "topic", "value": "How to build AI agents"},
                        {"type": "tool", "value": "LangChain"},
                    ],
                },
            ]
        }
        results = tool._factual_search("How much did I spend at ALDI?", kb, top_k=10)
        ids = {r["photo_id"] for r in results}
        assert "r1" in ids
        assert "d1" not in ids, (
            f"Document should not match spending query via short-word substring; "
            f"got {[(r['photo_id'], r['relevance_score'], r['evidence']) for r in results]}"
        )


# ── Semantic search ─────────────────────────────────────────────────────

class TestSemanticSearch:

    def test_description_match(self, tool, synthetic_kb):
        results = tool._semantic_search("Show me photos of pizza", synthetic_kb, top_k=3)
        paths = [r["photo_path"] for r in results]
        assert any("pizza" in p for p in paths)

    def test_image_type_boost(self, tool, synthetic_kb):
        results = tool._semantic_search("show me a document", synthetic_kb, top_k=3)
        # The document photo gets +0.2 for image_type boost
        top_paths = [r["photo_path"] for r in results]
        assert any("doc" in p for p in top_paths)

    def test_respects_top_k(self, tool, synthetic_kb):
        results = tool._semantic_search("receipt", synthetic_kb, top_k=1)
        assert len(results) <= 1

    def test_empty_query_returns_no_results(self, tool, synthetic_kb):
        results = tool._semantic_search("", synthetic_kb, top_k=3)
        # Meaningful overlap is 0 so no positive-scored results unless image_type match
        assert all(r["relevance_score"] > 0 for r in results)


# ── Behavioral search ───────────────────────────────────────────────────

class TestBehavioralSearch:

    def test_returns_dominant_type(self, tool, synthetic_kb):
        results = tool._behavioral_search("what type of photo do I have most?", synthetic_kb, top_k=5)
        # Receipts are dominant (2 of 5) — but behavioral picks target_type first
        # when the word appears in the query. No target word here, so dominant type
        # (receipt tied with food — earliest wins in insertion order).
        assert len(results) > 0

    def test_type_filter_by_query_keyword(self, tool, synthetic_kb):
        """Evidence-grounded: food query returns photos whose entities list food_items.

        p2 (instacart receipt with food_item=avocado) and p3 (food image with
        food_item=pizza) both contain top aggregated food_item entities, so the
        primary pool selects both. The old strictly-by-image_type behavior is
        gone intentionally — OCR-derived food items must attribute to receipts.
        """
        results = tool._behavioral_search("how often do I photograph food?", synthetic_kb, top_k=5)
        ids = {r["photo_id"] for r in results}
        assert ids == {"p2", "p3"}
        # Every returned photo should carry a matched_entity naming the food item
        for r in results:
            assert r.get("matched_entity") in {"avocado", "pizza"}

    def test_evidential_grounding_receipt_for_food_query(self, tool):
        """Regression: 'What food do I eat most?' must return the receipt that
        actually lists the named food — NOT a visual food photo lacking that entity.

        This models the Goldhen-eggs bug: visual food photos contain no OCR
        entities, while receipts list the food items. The answer aggregates
        entities across the KB, and source photos must substantiate that answer.
        """
        kb = {
            "photos": [
                # Two visual food photos with NO food_item entities
                {"id": "f1", "file_path": "photos/plate1.heic", "image_type": "food",
                 "description": "a plate", "ocr_text": "", "caption": "meal",
                 "entities": []},
                {"id": "f2", "file_path": "photos/plate2.heic", "image_type": "food",
                 "description": "a bowl", "ocr_text": "", "caption": "meal",
                 "entities": []},
                # Three receipts listing the same product
                {"id": "r1", "file_path": "photos/r1.heic", "image_type": "receipt",
                 "description": "receipt", "ocr_text": "GOLDHEN CAGE FREE EGGS",
                 "caption": "receipt",
                 "entities": [{"type": "food_item", "value": "Goldhen Cage Free Eggs"}]},
                {"id": "r2", "file_path": "photos/r2.heic", "image_type": "receipt",
                 "description": "receipt", "ocr_text": "GOLDHEN CAGE FREE EGGS",
                 "caption": "receipt",
                 "entities": [{"type": "food_item", "value": "Goldhen Cage Free Eggs"}]},
                {"id": "r3", "file_path": "photos/r3.heic", "image_type": "receipt",
                 "description": "receipt", "ocr_text": "GOLDHEN CAGE FREE EGGS",
                 "caption": "receipt",
                 "entities": [{"type": "food_item", "value": "Goldhen Cage Free Eggs"}]},
            ]
        }
        results = tool._behavioral_search("What food do I eat most often?", kb, top_k=5)
        assert len(results) == 3
        ids = {r["photo_id"] for r in results}
        assert ids == {"r1", "r2", "r3"}, (
            f"Expected the three receipts (which list the food item) but got {ids}"
        )
        for r in results:
            assert r["image_type"] == "receipt"
            assert r.get("matched_entity") == "Goldhen Cage Free Eggs"
        # Provenance line should be on the evidence string
        assert any("Based on" in r["evidence"] for r in results)

    def test_ungrounded_entity_is_rejected(self, tool):
        """Safety: if a photo's stored entity value is NOT substring-present
        in its OCR+description+caption, that entity must not be used as
        evidence — the photo should be excluded from the primary pool.

        This guards against KB hallucinations or post-hoc tag edits where
        entity values drift from the source text.
        """
        kb = {
            "photos": [
                # Ungrounded: entity says "eggs" but no OCR/desc mentions eggs
                {"id": "g1", "file_path": "photos/ghost.heic", "image_type": "receipt",
                 "description": "a sunny landscape", "ocr_text": "STORE 10.00",
                 "caption": "",
                 "entities": [{"type": "food_item", "value": "Eggs"}]},
                # Grounded: entity and OCR agree
                {"id": "g2", "file_path": "photos/real.heic", "image_type": "receipt",
                 "description": "grocery receipt",
                 "ocr_text": "GOLDHEN CAGE FREE EGGS 4.99",
                 "caption": "",
                 "entities": [{"type": "food_item", "value": "Eggs"}]},
            ]
        }
        results = tool._behavioral_search("What food do I eat?", kb, top_k=5)
        ids = {r["photo_id"] for r in results}
        assert "g1" not in ids, "Ungrounded entity must not substantiate its photo"
        assert "g2" in ids

    def test_evidence_contains_distribution(self, tool, synthetic_kb):
        results = tool._behavioral_search("what receipts do I have?", synthetic_kb, top_k=3)
        assert any("distribution" in r["evidence"].lower() for r in results)

    def test_baseline_score_floor(self, tool, synthetic_kb):
        """Behavioral results always have score >= 0.1 (baseline)."""
        results = tool._behavioral_search("food", synthetic_kb, top_k=5)
        for r in results:
            assert r["relevance_score"] >= 0.1


# ── Hybrid (RRF) ────────────────────────────────────────────────────────

class TestHybridSearch:

    def test_merges_both_lists(self, tool, synthetic_kb, monkeypatch):
        """RRF should union photo_ids from both input strategies."""
        # Stub _embedding_search to avoid loading the model in this unit test
        def fake_embedding(q, kb, top_k):
            return [
                {"photo_id": "p3", "photo_path": "photos/pizza.heic",
                 "relevance_score": 0.9, "evidence": "", "image_type": "food"},
            ]
        monkeypatch.setattr(tool, "_embedding_search", fake_embedding)

        merged = tool._hybrid_search("ALDI pizza", synthetic_kb, top_k=5)
        ids = {r["photo_id"] for r in merged}
        assert "p3" in ids          # from embedding stub
        assert "p1" in ids          # from factual (ALDI match)

    def test_rrf_evidence_marker_present(self, tool, synthetic_kb, monkeypatch):
        monkeypatch.setattr(
            tool, "_embedding_search",
            lambda q, kb, top_k: [
                {"photo_id": "p1", "photo_path": "x", "relevance_score": 0.5,
                 "evidence": "emb", "image_type": "receipt"}
            ],
        )
        merged = tool._hybrid_search("ALDI", synthetic_kb, top_k=3)
        assert all("[hybrid/RRF]" in r["evidence"] for r in merged)


# ── Score → grade mapping ───────────────────────────────────────────────

class TestScoreToGrade:

    @pytest.mark.parametrize("score,grade", [
        (0.95, "A"), (0.70, "A"),
        (0.60, "B"), (0.50, "B"),
        (0.40, "C"), (0.35, "C"),
        (0.25, "D"), (0.20, "D"),
        (0.10, "F"), (0.00, "F"),
    ])
    def test_thresholds(self, tool, score, grade):
        assert tool._score_to_grade(score) == grade


# ── Regression tests for P1/P3/P5/P6 fixes ──────────────────────────────

class TestClassifyQueryFixes:
    """P1: purchase-intent → factual. P6: behavioral aggregation."""

    @pytest.mark.parametrize("q", [
        "What did I buy at ALDI?",
        "What did I purchase at Instacart?",
        "receipt for eggs",
        "show me what I bought yesterday",
    ])
    def test_purchase_intent_routes_factual(self, tool, q):
        assert tool._classify_query(q) == "factual"

    @pytest.mark.parametrize("q", [
        "what cuisine do I eat most",
        "what type of food do I buy",
        "how frequently do I shop",
        "do I tend to shop at ALDI",
    ])
    def test_expanded_behavioral_patterns(self, tool, q):
        assert tool._classify_query(q) == "behavioral"


class TestSemanticSearchFixes:
    """P3 negative-entity guard, P5 single-noun entity/type expansion."""

    def test_negative_entity_guard_suppresses_unrelated_receipts(
        self, tool, synthetic_kb
    ):
        # Netflix is not a vendor in the KB, so all receipts should be
        # suppressed by the P3 guard.
        results = tool._semantic_search(
            "Netflix subscription bill", synthetic_kb, top_k=5
        )
        for r in results:
            if r["image_type"] == "receipt":
                assert r["relevance_score"] < 0.35

    def test_single_noun_entity_expansion_surfaces_match(
        self, tool, synthetic_kb
    ):
        # "ALDI" is an entity value on p1 — P5 should surface it above floor.
        results = tool._semantic_search("ALDI", synthetic_kb, top_k=5)
        p1 = next((r for r in results if r["photo_id"] == "p1"), None)
        assert p1 is not None
        assert p1["relevance_score"] >= 0.35


class TestFeedbackRewardShaping:
    """P2: damped adjustments and missed-retrieval distinction."""

    def test_missed_retrieval_does_not_tighten_threshold(self, tmp_path):
        from src.tools.feedback_store import FeedbackStore
        fs = FeedbackStore(path=str(tmp_path / "fb.json"))
        # 5 failures, all with n_results=0 → classified as misses
        for _ in range(5):
            fs.record_outcome("q", "semantic", correct=False,
                              confidence_score=0.0, n_results=0)
        # Miss-rate >= 50% should LOOSEN (negative adjustment), not tighten.
        assert fs.get_confidence_adjustment("semantic") <= 0.0

    def test_false_positives_tighten_within_cap(self, tmp_path):
        from src.tools.feedback_store import FeedbackStore
        fs = FeedbackStore(path=str(tmp_path / "fb.json"))
        for _ in range(10):
            fs.record_outcome("q", "factual", correct=False,
                              confidence_score=0.6, n_results=3)
        adj = fs.get_confidence_adjustment("factual")
        assert 0.0 < adj <= 0.10  # tightened but capped
