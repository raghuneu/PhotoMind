"""Expanded test cases: 36 new cases appended to the original 20.

Imports and re-exports the original TEST_CASES unchanged, then defines
EXPANDED_TEST_CASES (36 new cases) and ALL_TEST_CASES (56 total).

Generation notes:
  - Every expected_answer is traceable to photo_index.json KB content.
  - IMG_3103.HEIC excluded (confidence=0.3, unreliable source).
  - Ambiguous cases use expected_type that reflects CORRECT intent after
    disambiguation — they are labeled to show what an RL-trained router
    SHOULD do, not what the keyword router actually does.
  - should_decline=True cases verified to have zero KB evidence.

KB photo coverage added by these new cases (previously uncovered):
  IMG_1773.PNG   — screenshot (online shopping app / green beans)
  IMG_1783.HEIC  — other (UNIQLO tag, $59.90)
  IMG_2429.HEIC  — food (falafel/fries)
  IMG_2598.HEIC  — food (cheese pizza)
  IMG_3177.PNG   — receipt (ALDI digital receipt, eggs/cheese/milk)
  IMG_3178.PNG   — receipt (Instacart $21.15, sweet potatoes)
  IMG_3184.PNG   — receipt (ALDI refunds + replacements)
  IMG_3185.PNG   — receipt (ALDI spinach wraps/eggs/milk)
  IMG_3186.PNG   — receipt (Instacart avocado, $25.94 charged)
  IMG_3192.PNG   — receipt (ALDI sourdough/spinach/potatoes)
  IMG_3193.PNG   — receipt (Instacart mushrooms/cauliflower, $27.27)
  IMG_3197.PNG   — receipt (ALDI bread/eggs/tomato paste)
  IMG_3198.PNG   — receipt ($21.24 total)
  IMG_3274.HEIC  — receipt (College Convenience Feb 20, $11.98)
"""

from eval.test_cases import TEST_CASES  # noqa: F401 — re-exported unchanged

# ── Factual queries (7 new) ──────────────────────────────────────────────────
# These test specific entity/amount/date facts directly readable from OCR text.

EXPANDED_TEST_CASES = [
    {
        "query": "How much did I spend at Trader Joe's?",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": "37.59",
        "category": "factual",
        "notes": (
            "New entity: Trader Joe's total. OCR: 'Balance to pay $37.59'. "
            "Paraphrase variant of existing Patel Brothers / Tropical Foods factual pattern."
        ),
    },
    {
        "query": "What did I buy at Patel Brothers?",
        "expected_type": "factual",
        "expected_photo": "IMG_2880.HEIC",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Items-listing factual query for Patel Brothers. OCR lists masala powder, "
            "hing powder, sambar powder, rasam, diet coke, soya wadi, hot sev, hot mixture, "
            "sunfeast biscuits. No single expected_answer_contains — correctness = photo match."
        ),
    },
    {
        "query": "How much was my second College Convenience receipt?",
        "expected_type": "factual",
        "expected_photo": "IMG_3274.HEIC",
        "expected_answer_contains": "11.98",
        "category": "factual",
        "notes": (
            "Tests disambiguation between two College Convenience receipts (IMG_3040.HEIC $6.99 "
            "vs IMG_3274.HEIC $11.98). 'Second' implies Feb 20 visit. Covers IMG_3274.HEIC."
        ),
    },
    {
        "query": "How much did the Instacart order with mushrooms and cauliflower cost?",
        "expected_type": "factual",
        "expected_photo": "IMG_3193.PNG",
        "expected_answer_contains": "27.27",
        "category": "factual",
        "notes": (
            "Multi-entity factual: vendor=Instacart + items=mushrooms/cauliflower. "
            "OCR: 'Total $27.27'. Covers IMG_3193.PNG."
        ),
    },
    {
        "query": "What was the total on the grocery receipt with eggs, noodles, and tomatoes?",
        "expected_type": "factual",
        "expected_photo": "IMG_3198.PNG",
        "expected_answer_contains": "21.24",
        "category": "factual",
        "notes": (
            "Multi-item factual grounded in IMG_3198.PNG OCR (eggs, noodles, garlic, onions, "
            "tomatoes, peppers). Total $21.24. Covers IMG_3198.PNG."
        ),
    },
    {
        "query": "What is the price of the UNIQLO jeans in my photos?",
        "expected_type": "factual",
        "expected_photo": "IMG_1783.HEIC",
        "expected_answer_contains": "59.90",
        "category": "factual",
        "notes": (
            "Non-receipt factual: clothing tag. OCR clearly shows '$59.90'. "
            "Covers IMG_1783.HEIC (only non-food, non-receipt, non-document 'other' photo)."
        ),
    },
    {
        "query": "What was the Instacart order total for the avocado order?",
        "expected_type": "factual",
        "expected_photo": "IMG_3186.PNG",
        "expected_answer_contains": "25.94",
        "category": "factual",
        "notes": (
            "Instacart avocado receipt. OCR: 'Total charged $25.94' (original auth $30.23 "
            "but final after refund is $25.94). Covers IMG_3186.PNG."
        ),
    },

    # ── Semantic queries (6 new) ─────────────────────────────────────────────
    # These test visual/descriptive matching, not OCR entity extraction.

    {
        "query": "Show me photos of street food or takeaway food",
        "expected_type": "semantic",
        "expected_photo": "IMG_2429.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Visual match: falafel/fries in foil container is classic street-food presentation. "
            "Covers IMG_2429.HEIC. No OCR text — purely description-based retrieval."
        ),
    },
    {
        "query": "Find photos of Mexican or Latin food",
        "expected_type": "semantic",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Visual/entity match: Chipotle burrito bowl. Chipotle is a Mexican-inspired chain. "
            "Corona Extra (IMG_2140.HEIC) is also plausible secondary match."
        ),
    },
    {
        "query": "Show me a photo of clothing or fashion",
        "expected_type": "semantic",
        "expected_photo": "IMG_1783.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Visual/description match: UNIQLO clothing tag on jeans. Only clothing-related "
            "photo in the KB. Covers IMG_1783.HEIC from semantic angle."
        ),
    },
    {
        "query": "Find photos that look like they were taken indoors at a restaurant",
        "expected_type": "semantic",
        "expected_photo": "IMG_2598.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Scene-based semantic: IMG_2598.HEIC description mentions 'glass display' and "
            "'person behind the counter', suggesting an indoor pizza counter/restaurant setting."
        ),
    },
    {
        "query": "Show me photos related to productivity or software engineering",
        "expected_type": "semantic",
        "expected_photo": "IMG_3407.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Topic-based semantic. Document photo (IMG_3407.JPG) covers workflow orchestration, "
            "subagent strategy, task management — all software engineering productivity topics."
        ),
    },
    {
        "query": "Find something that looks like a social media screenshot",
        "expected_type": "semantic",
        "expected_photo": "IMG_3490.PNG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "IMG_3490.PNG described as 'social media reel by Make Boston Home' with OCR showing "
            "social media text. IMG_1773.PNG (shopping app screenshot) is also valid secondary. "
            "Primary match is IMG_3490.PNG due to description explicitly mentioning social media."
        ),
    },

    # ── Behavioral queries (6 new) ───────────────────────────────────────────
    # These require frequency aggregation or pattern analysis across the corpus.

    {
        "query": "How many times did I order from Instacart?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3178.PNG",
        "expected_answer_contains": None,
        "category": "behavioral",
        "notes": (
            "Frequency count behavioral: Instacart appears in IMG_3178.PNG, IMG_3186.PNG, "
            "IMG_3193.PNG = 3 orders. expected_photo is representative; answer should cite all 3."
        ),
    },
    {
        "query": "What grocery items do I buy most frequently?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "behavioral",
        "notes": (
            "Cross-receipt frequency behavioral. Eggs appear in IMG_1853.HEIC (3x), IMG_3177.PNG, "
            "IMG_3185.PNG, IMG_3197.PNG, IMG_3198.PNG, IMG_3442.HEIC — most recurrent item. "
            "ALDI is best representative photo."
        ),
    },
    {
        "query": "Do I usually pay with debit or credit card?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "behavioral",
        "notes": (
            "Payment pattern behavioral. Multiple receipts mention 'Debit', 'US DEBIT', "
            "contactless debit. Instacart receipts use Visa/ApplePay. Aggregate pattern "
            "leans debit/contactless."
        ),
    },
    {
        "query": "How much have I spent on groceries across all receipts?",
        "expected_type": "behavioral",
        "expected_photo": None,
        "expected_answer_contains": None,
        "category": "behavioral",
        "notes": (
            "Total aggregation behavioral across all receipt photos. Requires summing totals "
            "from IMG_1853.HEIC ($18.69), IMG_2880.HEIC ($29.47), IMG_3040.HEIC ($6.99), "
            "IMG_3178.PNG ($21.15), IMG_3186.PNG ($25.94), IMG_3193.PNG ($27.27), "
            "IMG_3198.PNG ($21.24), IMG_3274.HEIC ($11.98), IMG_3442.HEIC ($15.45), "
            "IMG_3491.HEIC ($37.59). No single photo is correct; aggregate query."
        ),
    },
    {
        "query": "What types of stores do I have receipts from?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "behavioral",
        "notes": (
            "Store-type aggregation behavioral. Stores: ALDI (grocery), Patel Brothers (Indian "
            "grocery), College Convenience (convenience), Tropical Foods (ethnic grocery), "
            "Trader Joe's (grocery), Instacart (delivery). expected_photo is representative ALDI."
        ),
    },
    {
        "query": "Which receipt had the highest total?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": "37.59",
        "category": "behavioral",
        "notes": (
            "Max-aggregation behavioral. Trader Joe's $37.59 (IMG_3491.HEIC) is the highest "
            "confirmed total across all receipts with clear totals. Patel Brothers $29.47 "
            "is second. expected_answer_contains verifiable."
        ),
    },

    # ── Edge cases / should-decline (6 new) ─────────────────────────────────
    # All verified to have zero KB content that could answer them.

    {
        "query": "What was my rent payment last month?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No housing/rent receipts in KB. Should decline. Tests that factual router "
            "does not hallucinate an answer from grocery receipts."
        ),
    },
    {
        "query": "Show me photos of my dog",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No animal photos in KB (all food, receipts, document, screenshot, clothing tag). "
            "Should decline."
        ),
    },
    {
        "query": "What is my bank account balance?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "Financial data not in KB. Receipts show transaction amounts, not account balances. "
            "Should decline rather than conflate transaction totals with balance."
        ),
    },
    {
        "query": "Show me photos from last Christmas",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No holiday/Christmas scene photos in KB. IMG_1853.HEIC date is 12/12/25 (not "
            "Christmas day). Should decline — no Christmas content."
        ),
    },
    {
        "query": "What medications do I take?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No medical/pharmacy receipts or photos in KB. Should decline."
        ),
    },
    {
        "query": "How much did I tip at restaurants?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No restaurant dine-in receipts in KB (all grocery/convenience store receipts). "
            "Food photos have no receipt data. Should decline — tip information absent."
        ),
    },

    # ── Ambiguous queries (11) ────────────────────────────────────────────────
    # Critical for demonstrating RL value over keyword rules.
    # expected_type = CORRECT intent after disambiguation.
    # notes field explains the routing confusion and correct resolution.

    {
        "query": "What stores do I have receipts from?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: contains 'receipt' (factual keyword) but asks about the SET of stores "
            "across the corpus — this is behavioral aggregation. Keyword router will likely "
            "misroute to factual and return a single receipt. RL should learn to route behavioral. "
            "Answer: ALDI, Patel Brothers, College Convenience, Tropical Foods, Trader Joe's, "
            "Instacart."
        ),
    },
    {
        "query": "Show me something I spent a lot on",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": "37.59",
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'show me' is a semantic keyword but the query needs factual retrieval "
            "(find highest-amount receipt). Keyword router likely routes semantic and returns "
            "a visually interesting photo instead of the highest-cost receipt. RL should route "
            "factual. Best answer: Trader Joe's $37.59."
        ),
    },
    {
        "query": "Find my most expensive purchase",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": "37.59",
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'find' is a semantic keyword, 'most' is a behavioral keyword, but the "
            "correct resolution requires factual max-retrieval. Keyword router faces a three-way "
            "signal conflict. RL should learn factual intent. Answer: Trader Joe's $37.59."
        ),
    },
    {
        "query": "Pictures of what I eat",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'pictures of' is semantic phrasing, but 'what I eat' implies habitual "
            "pattern (behavioral intent). Keyword router routes semantic and returns one food photo. "
            "RL should learn behavioral and return a pattern/summary across all food photos: "
            "Chipotle bowl, pizza (x2), beer, falafel, ice cream. expected_photo is representative."
        ),
    },
    {
        "query": "Do I shop at ALDI?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: yes/no question about a named vendor. 'ALDI' triggers factual entity "
            "matching, but the question is asking about shopping HABIT (behavioral). Keyword "
            "router routes factual and returns one ALDI receipt. RL should route behavioral and "
            "confirm pattern: 7+ ALDI receipts found. expected_photo is representative."
        ),
    },
    {
        "query": "How often do I order delivery?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3178.PNG",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'order' could trigger factual entity search, but 'how often' clearly "
            "signals frequency/behavioral. Instacart delivery appears in 3 receipts. "
            "Keyword router may go factual on 'order'. RL should route behavioral."
        ),
    },
    {
        "query": "What did I get from the grocery store?",
        "expected_type": "factual",
        "expected_photo": "IMG_3491.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'what did I get' is factual (specific items), but 'grocery store' is "
            "vague (many photos qualify). Could route semantic (visual search for grocery store "
            "scenes) or behavioral (aggregate items list) or factual (latest/most specific "
            "receipt). RL challenge: which receipt to prioritize? Trader Joe's is best documented "
            "with 11-item list in OCR. expected_photo reflects most item-rich single receipt."
        ),
    },
    {
        "query": "What food have I been eating lately?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'what food' implies factual entity, 'have I been eating lately' implies "
            "behavioral pattern over time. Keyword router may go factual. RL should go behavioral "
            "and summarize across food photos: burrito bowl, pizza (x2), Corona, falafel/fries, "
            "ice cream. expected_photo is representative first food photo."
        ),
    },
    {
        "query": "What is on my ALDI receipts?",
        "expected_type": "factual",
        "expected_photo": "IMG_1853.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'ALDI receipts' (plural) could trigger behavioral aggregation, but "
            "'what is on' requests item-level factual detail. Also tests multi-photo factual "
            "retrieval (7+ ALDI photos). Keyword router handles singular ALDI well but plural "
            "creates confusion. RL should route factual and synthesize across multiple ALDI "
            "receipts."
        ),
    },
    {
        "query": "Show me my Instacart orders",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3178.PNG",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'show me' is semantic, 'Instacart' is a factual entity, 'orders' (plural) "
            "suggests behavioral listing. Keyword router will likely go factual on 'Instacart' and "
            "return one receipt. RL should recognize 'show me ... orders' as behavioral — return "
            "all 3 Instacart receipts (IMG_3178.PNG, IMG_3186.PNG, IMG_3193.PNG)."
        ),
    },
    {
        "query": "How much does a Chipotle bowl cost?",
        "expected_type": "factual",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS + SHOULD DECLINE: 'Chipotle' entity triggers factual routing and "
            "IMG_1762.HEIC will match. But that photo is a food photo with no receipt/price OCR — "
            "only the Chipotle bag text. The KB cannot answer the PRICE of a Chipotle bowl. "
            "should_decline=True because retrieval will find the photo but cannot provide the "
            "price. Tests that the system does not hallucinate a price from a food photo."
        ),
    },
]

# ── Combined suite ────────────────────────────────────────────────────────────
ALL_TEST_CASES = TEST_CASES + EXPANDED_TEST_CASES
