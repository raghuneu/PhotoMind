"""Expanded test cases: 63 new cases appended to the original 20.

Imports and re-exports the original TEST_CASES unchanged, then defines
EXPANDED_TEST_CASES (63 new cases) and ALL_TEST_CASES (83 total).

Generation notes:
  - Every expected_answer is traceable to photo_index.json KB content.
  - IMG_3103.HEIC excluded (confidence=0.3, unreliable source).
  - IMG_2642.HEIC excluded from unique test targets (duplicate of IMG_2880.HEIC
    — same Patel Brothers receipt, same date and total).
  - Ambiguous cases use expected_type that reflects CORRECT intent after
    disambiguation — they are labeled to show what an RL-trained router
    SHOULD do, not what the keyword router actually does.
  - should_decline=True cases verified to have zero KB evidence.

KB photo coverage added by these new cases (previously uncovered):
  Round 1 (original 36 expanded cases):
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

  Round 2 (30 new cases for 29 newly-ingested photos):
    1a150901...JPG — food (dosa with chutneys)
    38eb54a0...JPG — food (biryani tray with nuts/mushrooms)
    5f205ac2...JPG — receipt (Fresh Central Grocery 2, $8.27)
    70b73fcd...JPG — food (taco with sauces)
    985857e2...JPG — receipt (Stop&Shop, $8.98, cash)
    9afd6c6c...JPG — food (chocolate chip bun)
    9b782fa4...JPG — food (cheeseburger at cash-only diner)
    IMG_1569.HEIC  — other (le drude wine bottle)
    IMG_1827.PNG   — screenshot (Air Jordan 1 Mid sneakers)
    IMG_1987.HEIC  — food (soju, Korean dining)
    IMG_2121.HEIC  — food (taco with cilantro/cheese)
    IMG_2896.HEIC  — food (black pepper grinder)
    IMG_3075.PNG   — screenshot (Building AI Agents resource list)
    IMG_3124.PNG   — screenshot (GenAI project structure)
    IMG_3200.PNG   — document (RAG Architectures infographic)
    IMG_3216.PNG   — screenshot (MCP vs RAG vs AI Agents)
    IMG_3217.PNG   — document (How to Build AI Agents steps)
    IMG_3551.HEIC  — receipt (College Convenience, $7.48)
    IMG_3570.PNG   — screenshot (Nawabi Hyderabad House Google Maps)
    IMG_3585.HEIC  — food (Pueblo Lindo flour tortillas package)
    a08f84f5...JPG — food (chicken wings with pickled vegetables)
    bab96f4f 2.jpg — receipt (Manorath Siddhi Grocery, $8.05)
    bab96f4f 3.jpg — receipt (99¢ Power, $9.76, cash)
    bab96f4f.jpg   — receipt (Lucky's Farm Inc, $41.40, cash)
    bb71bbb7...JPG — food (biryani bowl)
    d95520fb...JPG — food (Nathan's hot dog)
    f95fc698...JPG — bill (Stop&Shop $9.18 + Fresh Central Grocery $8.27)
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
        "expected_top_entity": "instacart",
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
        "expected_top_entity": "eggs",
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
        "expected_top_entity": "debit",
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
        "expected_top_entity": "aldi",
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
        "expected_top_entity": "aldi",
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
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg",
        "expected_answer_contains": "41.40",
        "expected_top_entity": "lucky",
        "category": "behavioral",
        "notes": (
            "Max-aggregation behavioral. Lucky's Farm $41.40 (bab96f4f...jpg) is the highest "
            "confirmed total across all receipts. Trader Joe's $37.59 is second, "
            "Patel Brothers $29.47 is third. expected_answer_contains verifiable."
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
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS + SHOULD DECLINE: 'Chipotle' entity triggers factual routing and "
            "IMG_1762.HEIC will match. But that photo is a food photo with no receipt/price OCR — "
            "only the Chipotle bag text. The KB cannot answer the PRICE of a Chipotle bowl. "
            "should_decline=True because retrieval will find the photo but cannot provide the "
            "price. Tests that the system does not hallucinate a price from a food photo. "
            "expected_photo=None: correct behavior is to decline, not to return the photo."
        ),
    },

    # ══════════════════════════════════════════════════════════════════════════
    # Round 2: 30 new test cases covering the 29 newly-ingested photos
    # ══════════════════════════════════════════════════════════════════════════

    # ── Factual queries (10 new) ─────────────────────────────────────────────

    {
        "query": "How much did I spend at Lucky's Farm?",
        "expected_type": "factual",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg",
        "expected_answer_contains": "41.40",
        "category": "factual",
        "notes": (
            "Lucky's Farm Inc receipt. OCR: 'TOTAL $41.40'. Cash payment. "
            "Covers bab96f4f...jpg. New vendor not in original test suite."
        ),
    },
    {
        "query": "What did I buy at Lucky's Farm?",
        "expected_type": "factual",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg",
        "expected_answer_contains": None,
        "category": "factual",
        "notes": (
            "Item-listing factual for Lucky's Farm. OCR lists: Capsicum x2 ($4.95), "
            "Tomato x20 ($22.70), Garlic ($3.00 x2), Ginger x4 ($7.75). "
            "Tests OCR entity extraction from a produce receipt."
        ),
    },
    {
        "query": "How much did I spend at Stop & Shop?",
        "expected_type": "factual",
        "expected_photo": "985857e2-dc26-46f3-9926-8497065ece59.JPG",
        "expected_answer_contains": "8.98",
        "category": "factual",
        "notes": (
            "Stop&Shop receipt factual. OCR: 'BALANCE 8.98', paid $10.00 cash, change $1.02. "
            "Covers 985857e2...JPG. Tests cash-payment receipt retrieval."
        ),
    },
    {
        "query": "How much did I spend at Manorath Siddhi Grocery?",
        "expected_type": "factual",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1 2.jpg",
        "expected_answer_contains": "8.05",
        "category": "factual",
        "notes": (
            "Manorath Siddhi Grocery receipt. OCR: 'TOTAL $8.05'. Purchased BADSHAH Chat "
            "Masala 500g. Covers bab96f4f... 2.jpg."
        ),
    },
    {
        "query": "How much was my 99 Cent Power receipt?",
        "expected_type": "factual",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1 3.jpg",
        "expected_answer_contains": "9.76",
        "category": "factual",
        "notes": (
            "99¢ Power receipt. OCR: 'TOTAL $9.76'. Cash payment. "
            "Covers bab96f4f... 3.jpg."
        ),
    },
    {
        "query": "How much did I spend at Fresh Central Grocery?",
        "expected_type": "factual",
        "expected_photo": "5f205ac2-2942-4f76-9709-03dfcdc0b7ad.JPG",
        "expected_answer_contains": "8.27",
        "category": "factual",
        "notes": (
            "Fresh Central Grocery 2 receipt. OCR: 'Total Sale $ 8.27'. Items include "
            "PARLE G GOLD, KCB MINERAL PAY, Lemons. Covers 5f205ac2...JPG."
        ),
    },
    {
        "query": "How much was my third College Convenience receipt?",
        "expected_type": "factual",
        "expected_photo": "IMG_3551.HEIC",
        "expected_answer_contains": "7.48",
        "category": "factual",
        "notes": (
            "Third College Convenience visit (April 13, 2026). OCR: 'Grand Total $7.48'. "
            "Items: Domino Sugar + MTR rosted vernicelli. Covers IMG_3551.HEIC. "
            "Tests disambiguation among 3 College Convenience receipts."
        ),
    },
    {
        "query": "What restaurant did I look up on Google Maps?",
        "expected_type": "factual",
        "expected_photo": "IMG_3570.PNG",
        "expected_answer_contains": "Nawabi",
        "category": "factual",
        "notes": (
            "Screenshot factual. OCR/description: 'Nawabi Hyderabad House' with 3.9 stars, "
            "Indian restaurant, $20-30 range. Covers IMG_3570.PNG."
        ),
    },

    # ── Semantic queries (8 new) ─────────────────────────────────────────────

    {
        "query": "Show me photos of Indian food",
        "expected_type": "semantic",
        "expected_photo": "1a150901-d289-41c7-9b77-23fb2546162f.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Visual/description match: dosa with chutneys and potato filling. "
            "Biryani photos (38eb54a0, bb71bbb7) are also valid secondary matches. "
            "Covers 1a150901...JPG."
        ),
    },
    {
        "query": "Find photos of tacos",
        "expected_type": "semantic",
        "expected_photo": "70b73fcd-173e-4db0-9b8c-b70081005277.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Food-item semantic match. Two taco photos: 70b73fcd...JPG and IMG_2121.HEIC. "
            "Both have 'taco' in entities/description. Covers 70b73fcd...JPG."
        ),
    },
    {
        "query": "Show me photos of sneakers or shoes",
        "expected_type": "semantic",
        "expected_photo": "IMG_1827.PNG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Screenshot semantic match. Air Jordan 1 Mid sneaker shopping screenshot. "
            "Only footwear photo in KB. Covers IMG_1827.PNG."
        ),
    },
    {
        "query": "Find photos of biryani or rice dishes",
        "expected_type": "semantic",
        "expected_photo": "38eb54a0-c956-430d-a359-014b439dec24.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Food-item semantic. Two biryani photos: 38eb54a0...JPG (tray with nuts) and "
            "bb71bbb7...JPG (bowl). Both have 'biryani' in entities. Covers 38eb54a0...JPG."
        ),
    },
    {
        "query": "Show me photos of wine or alcohol",
        "expected_type": "semantic",
        "expected_photo": "IMG_1569.HEIC",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Beverage semantic. IMG_1569.HEIC shows 'le drude' wine bottle. "
            "IMG_2140.HEIC (Corona beer) and IMG_1987.HEIC (soju) are secondary. "
            "Covers IMG_1569.HEIC."
        ),
    },
    {
        "query": "Find infographics about RAG or AI architectures",
        "expected_type": "semantic",
        "expected_photo": "IMG_3200.PNG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Topic-based semantic. IMG_3200.PNG is RAG Architectures infographic. "
            "IMG_3216.PNG (MCP vs RAG vs AI Agents) is also valid. "
            "Covers IMG_3200.PNG and IMG_3216.PNG."
        ),
    },
    {
        "query": "Show me a photo from a fast food restaurant",
        "expected_type": "semantic",
        "expected_photo": "d95520fb-0e5f-4e55-9cee-44ce7d3de613.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Scene-based semantic. Nathan's hot dog photo — interior of fast food restaurant. "
            "9b782fa4...JPG (cheeseburger at diner) is also valid secondary. "
            "Covers d95520fb...JPG."
        ),
    },
    {
        "query": "Find a photo of chicken or meat",
        "expected_type": "semantic",
        "expected_photo": "a08f84f5-cf93-4704-8d25-fbb3c27d2ff8.JPG",
        "expected_answer_contains": None,
        "category": "semantic",
        "notes": (
            "Food-item semantic. Chicken wings with onions and pickled vegetables. "
            "Only clearly meat-focused photo in KB. Covers a08f84f5...JPG."
        ),
    },

    # ── Behavioral queries (5 new) ───────────────────────────────────────────

    {
        "query": "How many different stores in Jersey City do I have receipts from?",
        "expected_type": "behavioral",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg",
        "expected_answer_contains": None,
        "expected_top_entity": "jersey city",
        "category": "behavioral",
        "notes": (
            "Location-based frequency behavioral. Jersey City stores: Patel Brothers, "
            "Lucky's Farm, 99¢ Power, Manorath Siddhi Grocery, Fresh Central Grocery, "
            "Stop&Shop. At least 6 unique stores. Tests geographic aggregation."
        ),
    },
    {
        "query": "Do I ever pay with cash?",
        "expected_type": "behavioral",
        "expected_photo": "985857e2-dc26-46f3-9926-8497065ece59.JPG",
        "expected_answer_contains": None,
        "expected_top_entity": "cash",
        "category": "behavioral",
        "notes": (
            "Payment method behavioral. Cash payments: Stop&Shop ($8.98), 99¢ Power ($9.76), "
            "Lucky's Farm ($41.40). Three cash receipts among ~22 total. "
            "Tests payment pattern detection across heterogeneous receipts."
        ),
    },
    {
        "query": "How many food photos do I have?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_1762.HEIC",
        "expected_answer_contains": None,
        "expected_top_entity": "food",
        "category": "behavioral",
        "notes": (
            "Count behavioral. 16 photos with image_type='food' in the 53-photo KB. "
            "Tests corpus-wide type counting. expected_photo is representative."
        ),
    },
    {
        "query": "What cuisines do I eat based on my photos?",
        "expected_type": "behavioral",
        "expected_photo": "1a150901-d289-41c7-9b77-23fb2546162f.JPG",
        "expected_answer_contains": None,
        "expected_top_entity": "food",
        "category": "behavioral",
        "notes": (
            "Cuisine diversity behavioral. Photos cover: Indian (dosa, biryani), Mexican "
            "(tacos, Chipotle), Italian (pizza), American (hot dog, cheeseburger), "
            "Middle Eastern (falafel), Korean (soju). Tests cross-cuisine aggregation."
        ),
    },
    {
        "query": "How many College Convenience receipts do I have?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3040.HEIC",
        "expected_answer_contains": None,
        "expected_top_entity": "college convenience",
        "category": "behavioral",
        "notes": (
            "Store frequency count. Three College Convenience receipts: IMG_3040.HEIC ($6.99), "
            "IMG_3274.HEIC ($11.98), IMG_3551.HEIC ($7.48). Tests vendor-specific counting."
        ),
    },

    # ── Edge cases / should-decline (3 new) ─────────────────────────────────

    {
        "query": "Show me photos of my car",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No vehicle photos in KB. All photos are food, receipts, documents, "
            "screenshots, or product tags. Should decline."
        ),
    },
    {
        "query": "What was my Amazon order?",
        "expected_type": "factual",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No Amazon receipts or order confirmations in KB. Should decline. "
            "Tests that the system doesn't confuse Instacart/ALDI online orders with Amazon."
        ),
    },
    {
        "query": "Show me selfies or portraits",
        "expected_type": "semantic",
        "expected_photo": None,
        "expected_answer_contains": None,
        "should_decline": True,
        "category": "edge_case",
        "notes": (
            "No portrait or selfie photos in KB. Some photos show hands holding items "
            "but no faces. Should decline."
        ),
    },

    # ── Ambiguous queries (4 new) ────────────────────────────────────────────

    {
        "query": "What spices do I buy?",
        "expected_type": "behavioral",
        "expected_photo": "IMG_2880.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'what spices' is factual entity extraction, but 'do I buy' implies "
            "habitual pattern (behavioral). Spices appear in Patel Brothers receipts (masala, "
            "hing, sambar powder, rasam, garam masala), Lucky's Farm (ginger), and "
            "Manorath Siddhi (Chat Masala). RL should route behavioral for cross-receipt "
            "aggregation."
        ),
    },
    {
        "query": "Show me what I bought in Jersey City",
        "expected_type": "behavioral",
        "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'show me' is semantic, 'Jersey City' is a factual entity, but "
            "'what I bought' across a location implies behavioral aggregation. Multiple "
            "Jersey City receipts: Patel Brothers, Lucky's Farm, Stop&Shop, 99¢ Power, "
            "Fresh Central Grocery, Manorath Siddhi. RL should aggregate."
        ),
    },
    {
        "query": "Find things I bought at convenience stores",
        "expected_type": "behavioral",
        "expected_photo": "IMG_3040.HEIC",
        "expected_answer_contains": None,
        "category": "ambiguous",
        "notes": (
            "AMBIGUOUS: 'find' is semantic keyword, 'things I bought' is factual, but "
            "'convenience stores' (plural) implies behavioral aggregation. College Convenience "
            "appears 3 times. RL should aggregate across all convenience store receipts."
        ),
    },
]

# ── Combined suite ────────────────────────────────────────────────────────────
ALL_TEST_CASES = TEST_CASES + EXPANDED_TEST_CASES

# ── Train / Held-Out Split ───────────────────────────────────────────────────
# Deterministic split: hold out ~25% (21 of 83 queries) for generalization eval.
# Selection: 4 factual, 4 semantic, 4 behavioral, 4 edge_case, 5 ambiguous.
# Chosen to cover each category while keeping the majority for training.
# Includes queries from both Round 1 and Round 2 test cases.
# Note: the National Grid gas bill factual held-out query was removed along with
# its source photo (PII redaction), reducing held-out from 22 -> 21 and train
# from 64 -> 62 (total 83 = 20 original + 63 expanded after -3 bill cases).

_HELD_OUT_QUERIES = frozenset({
    # Factual (4): original + round1 + round2
    "How much did I pay at Tropical Foods?",
    "How much did I spend at Trader Joe's?",
    "What was the Instacart order total for the avocado order?",
    "How much was my 99 Cent Power receipt?",
    # Semantic (4): original + round1 + round2
    "Find photos of beer or drinks",
    "Find photos that look like they were taken indoors at a restaurant",
    "Find photos of biryani or rice dishes",
    "Show me photos of sneakers or shoes",
    # Behavioral (4): original + round1 + round2
    "Which store do I shop at most often?",
    "Which receipt had the highest total?",
    "Do I ever pay with cash?",
    "What cuisines do I eat based on my photos?",
    # Edge case (4): original + round1 + round2
    "Show me photos from Paris",
    "Show me photos of my dog",
    "How much did I tip at restaurants?",
    "What was my Amazon order?",
    # Ambiguous (5): critical for RL generalization test
    "Find my most expensive purchase",
    "Do I shop at ALDI?",
    "What food have I been eating lately?",
    "How much does a Chipotle bowl cost?",
    "Show me what I bought in Jersey City",
})

HELD_OUT_TEST_CASES = [tc for tc in ALL_TEST_CASES if tc["query"] in _HELD_OUT_QUERIES]
TRAIN_TEST_CASES = [tc for tc in ALL_TEST_CASES if tc["query"] not in _HELD_OUT_QUERIES]

assert len(HELD_OUT_TEST_CASES) == 21, f"Expected 21 held-out, got {len(HELD_OUT_TEST_CASES)}"
assert len(TRAIN_TEST_CASES) == 62, f"Expected 62 train, got {len(TRAIN_TEST_CASES)}"
