"""Novel held-out test cases — true out-of-distribution generalization check.

These 15 queries are hand-written with intent shifts that are NOT paraphrases of
queries in TEST_CASES or EXPANDED_TEST_CASES. They combine multiple axes of
reasoning (vendor + comparison, item + aggregate, etc.) to stress the RL router.

None of these queries (or close variants) appear in the training split. Every
expected_answer is traceable to photo_index.json OCR/description content.

Category breakdown:
  5 factual    — specific entities, amounts, or items via OCR extraction
  4 semantic   — descriptive/visual match, no OCR keywords present
  3 behavioral — cross-corpus aggregation / pattern reasoning
  3 edge cases — zero KB evidence, should decline
"""

# ── Factual (5) ────────────────────────────────────────────────────────────
# Each combines a vendor, item, or context dimension with a specific numeric
# or entity answer that can be verified against OCR text.

NOVEL_TEST_CASES = [
    {
        "query": "Which store charged me the most tax on a single receipt?",
        "expected_type": "factual",
        "expected_photo": "IMG_2880.HEIC",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Tax-max factual — novel axis (tax, not subtotal/total). Patel Brothers "
            "IMG_2880.HEIC lists tax lines in OCR. Intent-shift: 'most' reads like "
            "behavioral keyword but resolution is a single factual lookup over a "
            "narrow set of receipts with tax rows."
        ),
    },
    {
        "query": "What's the cheapest thing I bought on any Instacart order?",
        "expected_type": "factual",
        "expected_photo": "IMG_3178.PNG",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Min-price factual scoped to Instacart. Requires OCR line-item parsing "
            "across IMG_3178/IMG_3186/IMG_3193/IMG_3186.PNG. 'Cheapest' is a "
            "superlative that keyword routers often push to behavioral."
        ),
    },
    {
        "query": "On what date did I shop at Trader Joe's?",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Date-extraction factual. Trader Joe's receipt (IMG_3491.HEIC) OCR "
            "contains a purchase date. Tests OCR date-field retrieval rather than "
            "totals — a dimension not previously covered."
        ),
    },
    {
        "query": "What is the SKU or item number on the UNIQLO tag?",
        "expected_type": "factual",
        "expected_photo": "IMG_1783.HEIC",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Non-receipt factual. UNIQLO tag OCR contains product code text — "
            "distinct from the $59.90 price covered in expanded cases. Tests that "
            "retrieval returns the tag photo even when the query targets a minor "
            "OCR field."
        ),
    },
    {
        "query": "Which receipt mentions a refund?",
        "expected_type": "factual",
        "expected_photo": "IMG_3184.PNG",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Novel lexical angle — 'refund' keyword appears in IMG_3184.PNG (ALDI "
            "refunds + replacements receipt). No other receipt uses the word. "
            "Tests specific-token retrieval against a single-match target."
        ),
    },

    # ── Semantic (4) ───────────────────────────────────────────────────────
    # Purely descriptive; target photos contain no keyword overlap with the query.

    {
        "query": "Find a picture that feels cozy or warm",
        "expected_type": "semantic",
        "expected_photo": "IMG_2598.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Affective/mood-based semantic. Pizza counter scene (warm lighting, "
            "food, indoor setting) is the closest match. No OCR keywords will help "
            "— requires embedding similarity on description."
        ),
    },
    {
        "query": "Show me something with bright colors",
        "expected_type": "semantic",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Visual attribute semantic. Chipotle bowl photo has vivid food colors "
            "(rice, salsa, greens). Falafel (IMG_2429) also plausible secondary. "
            "Tests description-based color retrieval."
        ),
    },
    {
        "query": "Find a photo that looks like a handwritten note or scribble",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "semantic",
        "notes": (
            "Edge-semantic: no handwritten content in KB (all printed receipts, "
            "printed tags, typed documents, food photos). Tests that semantic "
            "search declines rather than forcing a spurious match."
        ),
    },
    {
        "query": "Which photo shows the most items packed together?",
        "expected_type": "semantic",
        "expected_photo": "IMG_2880.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Density-based semantic. Patel Brothers receipt has the longest item "
            "list visible. Intent-shift: 'most items' reads behavioral but answer "
            "is a single visual match on item density."
        ),
    },

    # ── Behavioral (3) ─────────────────────────────────────────────────────
    # Require aggregation / comparison across the full corpus.

    {
        "query": "Do I spend more at ALDI or at Instacart on average?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "expected_top_entity": "aldi",
        "category": "behavioral",
        "notes": (
            "Cross-vendor comparison behavioral. Requires computing per-vendor "
            "average totals across multiple receipts. ALDI receipts tend to be "
            "smaller totals; Instacart tends higher. No single photo answers it."
        ),
    },
    {
        "query": "What percentage of my photos are grocery receipts?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "expected_top_entity": "receipt",
        "category": "behavioral",
        "notes": (
            "Proportion/distribution behavioral. Requires counting receipts vs "
            "total KB size. ~22 receipts + 2 bills out of 53 photos (~41-45%). "
            "Pure aggregate question — expected_photo is a representative receipt."
        ),
    },
    {
        "query": "Am I eating more home-cooked meals or takeout based on my photos?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "expected_top_entity": "food",
        "category": "behavioral",
        "notes": (
            "Pattern-inference behavioral. Requires classifying food photos as "
            "takeout (Chipotle, pizza, falafel, tacos, hot dog, cheeseburger) vs "
            "home-cooked (tortillas package, black pepper grinder suggest some). "
            "16 food photos total. Intent-shift: keyword router may seize on "
            "'eating' and route factual."
        ),
    },

    # ── Edge cases (3) ─────────────────────────────────────────────────────
    # Zero KB evidence — should decline.

    {
        "query": "What airline did I fly with on my last trip?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No boarding passes, airline tickets, or travel documents in KB. "
            "Should decline cleanly without fabricating an airline."
        ),
    },
    {
        "query": "How many steps did I walk today?",
        "expected_type": "behavioral",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No fitness/health tracker screenshots in KB. Should decline rather "
            "than confabulating a number from unrelated OCR digits."
        ),
    },
    {
        "query": "What was the weather the day I went to Patel Brothers?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "Patel Brothers receipt exists (IMG_2880.HEIC) but no weather data "
            "lives in the KB. Tricky decline because the vendor IS known — tests "
            "that 'no-data' decision is fact-scoped, not photo-scoped."
        ),
    },
]

assert len(NOVEL_TEST_CASES) == 15, f"Expected 15 novel cases, got {len(NOVEL_TEST_CASES)}"

_by_cat = {}
for _tc in NOVEL_TEST_CASES:
    _by_cat[_tc["category"]] = _by_cat.get(_tc["category"], 0) + 1
assert _by_cat == {"factual": 5, "semantic": 4, "behavioral": 3, "edge_case": 3}, (
    f"Unexpected category distribution: {_by_cat}"
)
