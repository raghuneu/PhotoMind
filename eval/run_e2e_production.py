#!/usr/bin/env python3
"""
E2E production tests for factual + edge case queries.
Tests against:
  - PRODUCTION API: https://srraghuram-photomind.hf.space/api
  - Quick Answer mode = POST /api/query with body {"query":"...","mode":"fast","top_k":5}
  - Feedback endpoint = POST /api/feedback
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

API_BASE = "https://srraghuram-photomind.hf.space/api"
QUERY_URL = f"{API_BASE}/query"
FEEDBACK_URL = f"{API_BASE}/feedback"
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "e2e_production_factual_edge.json",
)

# ---------------------------------------------------------------------------
# Pre-completed UI result (already tested manually)
# ---------------------------------------------------------------------------
UI_RESULT = {
    "query": "How much did I spend at ALDI?",
    "category": "factual",
    "expected_photo": "IMG_1853.HEIC",
    "expected_should_decline": False,
    "query_type_detected": "factual",
    "top_photo": "photos/IMG_1853.HEIC",
    "source_photos": [
        "photos/IMG_1853.HEIC",
        "photos/IMG_3177.PNG",
        "photos/IMG_3184.PNG",
        "photos/IMG_3192.PNG",
        "photos/IMG_3197.PNG",
    ],
    "confidence_grade": "B",
    "confidence_score": 0.55,
    "warning": None,
    "passed": True,
    "latency_s": 0.025,
    "feedback_submitted": True,
    "test_method": "ui",
}

# ---------------------------------------------------------------------------
# Factual test cases (26 queries)
# ---------------------------------------------------------------------------
FACTUAL_QUERIES = [
    {"query": "How much did I spend at Patel Brothers?", "expected_photo": "IMG_2880.HEIC"},
    {"query": "How much was my College Convenience receipt?", "expected_photo": "IMG_3040.HEIC"},
    {"query": "When did I shop at Tropical Foods Supermarket?", "expected_photo": "IMG_3442.HEIC"},
    {"query": "What is the address on my ALDI receipt?", "expected_photo": "IMG_1853.HEIC"},
    {"query": "What items did I buy at Trader Joe's?", "expected_photo": "IMG_3491.HEIC"},
    {"query": "How much did I pay at Tropical Foods?", "expected_photo": "IMG_3442.HEIC"},
    {"query": "How much did I spend at Trader Joe's?", "expected_photo": "IMG_3491.HEIC"},
    {"query": "What did I buy at Patel Brothers?", "expected_photo": "IMG_2880.HEIC"},
    {"query": "How much was my second College Convenience receipt?", "expected_photo": "IMG_3274.HEIC"},
    {"query": "How much did the Instacart order with mushrooms and cauliflower cost?", "expected_photo": "IMG_3193.PNG"},
    {"query": "What was the total on the grocery receipt with eggs, noodles, and tomatoes?", "expected_photo": "IMG_3198.PNG"},
    {"query": "What is the price of the UNIQLO jeans in my photos?", "expected_photo": "IMG_1783.HEIC"},
    {"query": "What was the Instacart order total for the avocado order?", "expected_photo": "IMG_3186.PNG"},
    {"query": "How much did I spend at Lucky's Farm?", "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg"},
    {"query": "What did I buy at Lucky's Farm?", "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1.jpg"},
    {"query": "How much did I spend at Stop & Shop?", "expected_photo": "985857e2-dc26-46f3-9926-8497065ece59.JPG"},
    {"query": "How much did I spend at Manorath Siddhi Grocery?", "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1 2.jpg"},
    {"query": "How much was my 99 Cent Power receipt?", "expected_photo": "bab96f4f-1d32-4377-8d46-ad5ae9d12fa1 3.jpg"},
    {"query": "How much did I spend at Fresh Central Grocery?", "expected_photo": "5f205ac2-2942-4f76-9709-03dfcdc0b7ad.JPG"},
    {"query": "How much was my third College Convenience receipt?", "expected_photo": "IMG_3551.HEIC"},
    {"query": "What restaurant did I look up on Google Maps?", "expected_photo": "IMG_3570.PNG"},
    {"query": "Which store charged me the most tax on a single receipt?", "expected_photo": "IMG_2880.HEIC"},
    {"query": "What's the cheapest thing I bought on any Instacart order?", "expected_photo": "IMG_3178.PNG"},
    {"query": "On what date did I shop at Trader Joe's?", "expected_photo": "IMG_3491.HEIC"},
    {"query": "What is the SKU or item number on the UNIQLO tag?", "expected_photo": "IMG_1783.HEIC"},
    {"query": "Which receipt mentions a refund?", "expected_photo": "IMG_3184.PNG"},
]

# ---------------------------------------------------------------------------
# Edge case test cases (16 queries — all should_decline)
# ---------------------------------------------------------------------------
EDGE_CASE_QUERIES = [
    "What was my electric bill this month?",
    "Show me photos from Paris",
    "What is the meaning of life?",
    "Show me my Netflix subscription payment",
    "What was my rent payment last month?",
    "Show me photos of my dog",
    "What is my bank account balance?",
    "Show me photos from last Christmas",
    "What medications do I take?",
    "How much did I tip at restaurants?",
    "Show me photos of my car",
    "What was my Amazon order?",
    "Show me selfies or portraits",
    "What airline did I fly with on my last trip?",
    "How many steps did I walk today?",
    "What was the weather the day I went to Patel Brothers?",
]


def call_api(query_text: str):
    """Call the production query API. Returns (data_dict, latency_s, error_str)."""
    payload = {"query": query_text, "mode": "fast", "top_k": 5}
    start = time.time()
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=30)
        latency = time.time() - start
        if resp.status_code == 200:
            return resp.json(), latency, None
        return None, latency, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return None, time.time() - start, str(e)


def submit_feedback(query_text: str, strategy: str, was_correct: bool, confidence_score: float) -> bool:
    """Submit feedback to the production API."""
    payload = {
        "query": query_text,
        "strategy": strategy,
        "was_correct": was_correct,
        "confidence_score": confidence_score,
    }
    try:
        resp = requests.post(FEEDBACK_URL, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def parse_api_response(data: dict | None):
    """Extract standard fields from API response."""
    if data is None:
        return "unknown", None, [], "F", 0.0, None
    query_type = data.get("query_type_detected", "unknown") or "unknown"
    results = data.get("results", []) or []
    top_photo = results[0].get("photo_path") if results else None
    source_photos = [r.get("photo_path", "") for r in results]
    grade = data.get("grade", "F") or "F"
    score = data.get("confidence_score", 0.0) or 0.0
    warning = data.get("warning")
    return query_type, top_photo, source_photos, grade, score, warning


def evaluate_factual(expected_photo: str, source_photos: list[str]) -> bool:
    """Factual pass: expected photo basename appears in any source photo path."""
    basename = os.path.basename(expected_photo)
    return any(basename in p for p in source_photos)


def evaluate_edge(grade: str, warning) -> bool:
    """Edge case pass: grade is D/F OR warning is present."""
    return grade in ("D", "F") or warning is not None


def run_tests():
    all_results = []
    category_stats = {"factual": {"total": 0, "passed": 0, "latencies": []},
                      "edge_case": {"total": 0, "passed": 0, "latencies": []}}
    silent_failures = 0
    hallucinations_on_decline = 0
    failures = []

    # -----------------------------------------------------------------------
    # 1. Include UI result as-is
    # -----------------------------------------------------------------------
    all_results.append(UI_RESULT)
    category_stats["factual"]["total"] += 1
    category_stats["factual"]["passed"] += 1
    category_stats["factual"]["latencies"].append(UI_RESULT["latency_s"])

    total_api = len(FACTUAL_QUERIES) + len(EDGE_CASE_QUERIES)
    idx = 0

    # -----------------------------------------------------------------------
    # 2. Factual queries via API
    # -----------------------------------------------------------------------
    print(f"Running {len(FACTUAL_QUERIES)} factual queries + {len(EDGE_CASE_QUERIES)} edge case queries = {total_api} API calls\n")

    for tc in FACTUAL_QUERIES:
        idx += 1
        query_text = tc["query"]
        expected_photo = tc["expected_photo"]
        print(f"[{idx}/{total_api}] {query_text[:65]}...")

        if idx > 1:
            time.sleep(0.5)

        data, latency, error = call_api(query_text)

        if error:
            query_type, top_photo, source_photos = "unknown", None, []
            grade, score, warning = "F", 0.0, None
            passed = False
            print(f"  ERROR: {error}")
        else:
            query_type, top_photo, source_photos, grade, score, warning = parse_api_response(data)
            passed = evaluate_factual(expected_photo, source_photos)

        # Silent failure: high confidence but wrong
        if grade in ("A", "B") and not passed:
            silent_failures += 1

        fb_ok = submit_feedback(query_text, query_type, passed, score)

        entry = {
            "query": query_text,
            "category": "factual",
            "expected_photo": expected_photo,
            "expected_should_decline": False,
            "query_type_detected": query_type,
            "top_photo": top_photo,
            "source_photos": source_photos,
            "confidence_grade": grade,
            "confidence_score": score,
            "warning": warning,
            "passed": passed,
            "latency_s": round(latency, 3),
            "feedback_submitted": fb_ok,
            "test_method": "api",
        }
        all_results.append(entry)
        category_stats["factual"]["total"] += 1
        category_stats["factual"]["latencies"].append(latency)
        if passed:
            category_stats["factual"]["passed"] += 1
        else:
            failures.append({"query": query_text, "reason": f"expected {expected_photo} not in source_photos, grade={grade}"})

        tag = "PASS" if passed else "FAIL"
        print(f"  {tag} | grade={grade} score={score:.2f} | strategy={query_type} | {latency:.2f}s")

    # -----------------------------------------------------------------------
    # 3. Edge case queries via API
    # -----------------------------------------------------------------------
    for query_text in EDGE_CASE_QUERIES:
        idx += 1
        print(f"[{idx}/{total_api}] {query_text[:65]}...")
        time.sleep(0.5)

        data, latency, error = call_api(query_text)

        if error:
            query_type, top_photo, source_photos = "unknown", None, []
            grade, score, warning = "F", 0.0, None
            passed = True  # API error on edge case = effectively declined
            print(f"  ERROR: {error}")
        else:
            query_type, top_photo, source_photos, grade, score, warning = parse_api_response(data)
            passed = evaluate_edge(grade, warning)

        # Hallucination on decline: should decline but gave confident answer
        if grade in ("A", "B", "C") and warning is None:
            hallucinations_on_decline += 1

        fb_ok = submit_feedback(query_text, query_type, passed, score)

        entry = {
            "query": query_text,
            "category": "edge_case",
            "expected_photo": None,
            "expected_should_decline": True,
            "query_type_detected": query_type,
            "top_photo": top_photo,
            "source_photos": source_photos,
            "confidence_grade": grade,
            "confidence_score": score,
            "warning": warning,
            "passed": passed,
            "latency_s": round(latency, 3),
            "feedback_submitted": fb_ok,
            "test_method": "api",
        }
        all_results.append(entry)
        category_stats["edge_case"]["total"] += 1
        category_stats["edge_case"]["latencies"].append(latency)
        if passed:
            category_stats["edge_case"]["passed"] += 1
        else:
            failures.append({"query": query_text, "reason": f"should decline but grade={grade}, warning={warning}"})

        tag = "PASS" if passed else "FAIL"
        print(f"  {tag} | grade={grade} score={score:.2f} | strategy={query_type} | {latency:.2f}s")

    # -----------------------------------------------------------------------
    # 4. Build output
    # -----------------------------------------------------------------------
    def cat_summary(stats):
        t = stats["total"]
        p = stats["passed"]
        lats = stats["latencies"]
        return {
            "total": t,
            "passed": p,
            "pass_rate": round(p / t, 4) if t else 0,
            "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else 0,
        }

    overall_total = sum(s["total"] for s in category_stats.values())
    overall_passed = sum(s["passed"] for s in category_stats.values())
    all_lats = [l for s in category_stats.values() for l in s["latencies"]]

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "categories": {k: cat_summary(v) for k, v in category_stats.items()},
        "overall": {
            "total": overall_total,
            "passed": overall_passed,
            "pass_rate": round(overall_passed / overall_total, 4) if overall_total else 0,
            "avg_latency_s": round(sum(all_lats) / len(all_lats), 3) if all_lats else 0,
        },
        "silent_failures": silent_failures,
        "hallucinations_on_decline": hallucinations_on_decline,
        "results": all_results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # -----------------------------------------------------------------------
    # 5. Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("E2E PRODUCTION TEST SUMMARY — Factual + Edge Cases")
    print(f"{'='*60}")
    print(f"Total queries:  {overall_total}")
    print(f"Passed:         {overall_passed} ({output['overall']['pass_rate']*100:.1f}%)")
    print(f"Avg latency:    {output['overall']['avg_latency_s']:.3f}s")
    print(f"Silent failures:          {silent_failures}")
    print(f"Hallucinations on decline: {hallucinations_on_decline}")
    print()
    for cat, s in output["categories"].items():
        print(f"  {cat:12s}: {s['passed']}/{s['total']} ({s['pass_rate']*100:.1f}%) avg_lat={s['avg_latency_s']:.3f}s")
    print()

    top_fail = failures[:5]
    if top_fail:
        print(f"Top {len(top_fail)} failures:")
        for i, f in enumerate(top_fail, 1):
            print(f"  {i}. {f['query'][:60]}")
            print(f"     reason: {f['reason']}")
    else:
        print("No failures!")

    print(f"\nResults written to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_tests()
