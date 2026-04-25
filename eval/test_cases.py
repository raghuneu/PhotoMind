"""Hand-labeled test queries with ground truth for evaluation.

All expected_photo values match actual filenames in knowledge_base/photo_index.json.
Photo contents (from ingestion):
  IMG_1762.HEIC  — food (Chipotle burrito bowl)
  IMG_1773.PNG   — screenshot (online shopping app)
  IMG_1783.HEIC  — other (UNIQLO clothing tag, $59.90)
  IMG_1853.HEIC  — receipt (ALDI Medford MA, $18.69, 12/12/25)
  IMG_2137.HEIC  — food (pizza slice)
  IMG_2140.HEIC  — food (Corona Extra beer)
  IMG_2429.HEIC  — food (falafel/fries in foil container)
  IMG_2598.HEIC  — food (cheese pizza)
  IMG_2880.HEIC  — receipt (Patel Brothers Jersey City NJ, $29.47, 1/3/26)
  IMG_3040.HEIC  — receipt (College Convenience Boston MA, $6.99, Jan 28 2026)
  IMG_3103.HEIC  — other (low confidence)
  IMG_3177.PNG   — receipt (ALDI groceries)
  IMG_3178.PNG   — receipt (Instacart, sweet potatoes)
  IMG_3184.PNG   — receipt (ALDI refunds/purchases)
  IMG_3185.PNG   — receipt (ALDI groceries, spinach wraps/eggs/milk)
  IMG_3186.PNG   — receipt (Instacart, avocado, $30.23)
  IMG_3192.PNG   — receipt (grocery items)
  IMG_3193.PNG   — receipt (Instacart, mushrooms/cauliflower, $27.27)
  IMG_3197.PNG   — receipt (ALDI, bread/eggs/tomato paste)
  IMG_3198.PNG   — receipt ($21.24 total)
  IMG_3274.HEIC  — receipt (College Convenience, $11.98, Feb 20 2026)
  IMG_3407.JPG   — document (workflow orchestration notes)
  IMG_3442.HEIC  — receipt (Tropical Foods Supermarket Roxbury MA, $15.45, Mar 23 2026)
  IMG_3490.PNG   — food (ice cream at Sullivan's Castle Island, $5.25)
  IMG_3491.HEIC  — receipt (Trader Joe's Boston MA, cracked wheat sourdough/garlic naan)
"""

TEST_CASES = [
    # ── Factual queries (7) ──────────────────────────────────────────
    {
        "query": "How much did I spend at ALDI?",
        "expected_type": "factual",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": "18.69",
        "category": "factual",
    },
    {
        "query": "How much did I spend at Patel Brothers?",
        "expected_type": "factual",
        "expected_photo": "IMG_2880.HEIC",
        "expected_answer_contains": "29.47",
        "category": "factual",
    },
    {
        "query": "How much was my College Convenience receipt?",
        "expected_type": "factual",
        "expected_photo": "IMG_3040.HEIC",
        "expected_answer_contains": "6.99",
        "category": "factual",
    },
    {
        "query": "When did I shop at Tropical Foods Supermarket?",
        "expected_type": "factual",
        "expected_photo": "IMG_3442.HEIC",
        "expected_answer_contains": "March",
        "category": "factual",
    },
    {
        "query": "What is the address on my ALDI receipt?",
        "expected_type": "factual",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": "Medford",
        "category": "factual",
    },
    {
        "query": "What items did I buy at Trader Joe's?",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": None,
        "category": "factual",
    },
    {
        "query": "How much did I pay at Tropical Foods?",
        "expected_type": "factual",
        "expected_photo": "IMG_3442.HEIC",
        "expected_answer_contains": "15.45",
        "category": "factual",
    },

    # ── Semantic queries (5) ─────────────────────────────────────────
    {
        "query": "Show me photos that feel like summer",
        "expected_type": "semantic",
        "expected_photo": "IMG_3490.PNG",
        "expected_answer_contains": None,
        "category": "semantic",
    },
    {
        "query": "Find a document about workflow or task management",
        "expected_type": "semantic",
        "expected_photo": "IMG_3407.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
    },
    {
        "query": "Show me photos of pizza",
        "expected_type": "semantic",
        "expected_photo": "IMG_2137.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
    },
    {
        "query": "Find photos of beer or drinks",
        "expected_type": "semantic",
        "expected_photo": "IMG_2140.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
    },
    {
        "query": "Show me outdoor or scenic photos",
        "expected_type": "semantic",
        "expected_photo": "IMG_3490.PNG",
        "expected_answer_contains": None,
        "category": "semantic",
    },

    # ── Behavioral queries (4) ───────────────────────────────────────
    {
        "query": "What type of food do I photograph most?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1762.HEIC",  # Representative food photo
        "expected_answer_contains": None,
        "expected_top_entity": "food",  # Outcome check: any food photo in results
        "category": "behavioral",
    },
    {
        "query": "How many receipts do I have?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",  # Representative receipt photo
        "expected_answer_contains": None,
        "expected_top_entity": "receipt",
        "category": "behavioral",
    },
    {
        "query": "What is the breakdown of my photo types?",
        "expected_type": "behavioral",
        "expected_photo": None,  # Aggregate query — any photo is valid
        "expected_answer_contains": None,
        "expected_top_entity": "receipt",  # Receipts are the dominant category
        "category": "behavioral",
    },
    {
        "query": "Which store do I shop at most often?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",  # ALDI receipt (most frequent store)
        "expected_answer_contains": None,
        "expected_top_entity": "aldi",
        "category": "behavioral",
    },

    # ── Edge cases / should-decline (4) ─────────────────────────────
    {
        "query": "What was my electric bill this month?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
    },
    {
        "query": "Show me photos from Paris",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
    },
    {
        "query": "What is the meaning of life?",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
    },
    {
        "query": "Show me my Netflix subscription payment",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
    },
]
