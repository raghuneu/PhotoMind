#!/usr/bin/env python3
"""
E2E production tests for behavioral + novel (behavioral/edge_case) queries.
Tests against:
  - PRODUCTION API: https://srraghuram-photomind.hf.space/api
  - Quick Answer mode = POST /api/query with body {"query":"...","mode":"fast","top_k":5}
  - Feedback endpoint = POST /api/feedback

Scope: behavioral queries from test_cases.py, expanded_test_cases.py (behavioral category),
       and novel_test_cases.py (behavioral + edge_case categories).
"""

import json
import time
import requests
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = "https://srraghuram-photomind.hf.space/api"
QUERY_URL = f"{API_BASE}/query"
FEEDBACK_URL = f"{API_BASE}/feedback"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "e2e_production_behavioral_novel.json")

# Collect all test cases
from eval.test_cases import TEST_CASES
from eval.expanded_test_cases import EXPANDED_TEST_CASES
from eval.novel_test_cases import NOVEL_TEST_CASES


def collect_queries():
    """Collect all behavioral + novel (behavioral/edge_case) queries."""
    queries = []

    # test_cases.py behavioral (4)
    for tc in TEST_CASES:
        if tc.get("category") == "behavioral":
            queries.append(tc)

    # expanded_test_cases.py behavioral (11)
    for tc in EXPANDED_TEST_CASES:
        if tc.get("category") == "behavioral":
            queries.append(tc)

    # novel_test_cases.py behavioral (3) + edge_case (3)
    for tc in NOVEL_TEST_CASES:
        if tc.get("category") in ("behavioral", "edge_case"):
            queries.append(tc)

    return queries


def call_api(query_text):
    """Call the production API with a query."""
    payload = {"query": query_text, "mode": "fast", "top_k": 5}
    start = time.time()
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=30)
        latency = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            return data, latency, None
        else:
            return None, latency, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        latency = time.time() - start
        return None, latency, str(e)


def submit_feedback(query_text, strategy, was_correct, confidence_score):
    """Submit feedback to the production API."""
    payload = {
        "query": query_text,
        "strategy": strategy,
        "was_correct": was_correct,
        "confidence_score": confidence_score
    }
    try:
        resp = requests.post(FEEDBACK_URL, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def evaluate_result(tc, data):
    """Evaluate whether a query result passes based on test case criteria."""
    if data is None:
        return False, "no_response"

    should_decline = tc.get("should_decline", False)
    expected_top_entity = tc.get("expected_top_entity")
    expected_photo = tc.get("expected_photo")
    expected_answer_contains = tc.get("expected_answer_contains")

    answer_summary = data.get("answer_summary", "") or ""
    results = data.get("results", []) or []
    grade = data.get("grade", "F") or "F"
    confidence = data.get("confidence_score", 0) or 0
    strategy = data.get("query_type_detected", "unknown") or "unknown"

    if should_decline:
        # For decline queries: pass if grade is D or F, or results empty
        passed = grade in ("D", "F") or len(results) == 0
        reason = "decline_check"
    elif expected_top_entity:
        # For aggregate queries: entity appears in answer or results
        entity_lower = expected_top_entity.lower()
        found_in_answer = entity_lower in answer_summary.lower()
        found_in_results = any(
            entity_lower in (r.get("image_type", "") or "").lower() or
            entity_lower in (r.get("evidence", "") or "").lower() or
            entity_lower in (r.get("description", "") or "").lower() or
            entity_lower in json.dumps(r.get("entities", []) or []).lower()
            for r in results
        )
        passed = (found_in_answer or found_in_results) and grade in ("A", "B", "C") and len(results) > 0
        reason = "entity_check"
    elif expected_answer_contains:
        found = expected_answer_contains.lower() in answer_summary.lower()
        passed = found and grade in ("A", "B", "C")
        reason = "answer_contains_check"
    elif expected_photo:
        photo_basename = os.path.basename(expected_photo)
        found = any(photo_basename in (r.get("photo_path", "") or "") for r in results)
        passed = found
        reason = "photo_match_check"
    else:
        # Generic aggregate: pass if grade A/B/C and results not empty
        passed = grade in ("A", "B", "C") and len(results) > 0
        reason = "aggregate_check"

    return passed, reason


def run_tests():
    """Run all behavioral + novel tests."""
    queries = collect_queries()
    print(f"Collected {len(queries)} queries for testing.")

    results = []
    total = len(queries)
    passed_count = 0
    total_latency = 0
    routing_errors = 0
    failures = []

    for i, tc in enumerate(queries):
        query_text = tc["query"]
        category = tc.get("category", "unknown")
        expected_type = tc.get("expected_type", "unknown")

        print(f"[{i+1}/{total}] Testing: {query_text[:60]}...")

        # Rate limit: 0.5s between requests
        if i > 0:
            time.sleep(0.5)

        data, latency, error = call_api(query_text)
        total_latency += latency

        if error:
            passed = False
            reason = f"api_error: {error}"
            strategy = "unknown"
            confidence = 0
            grade = "F"
        else:
            passed, reason = evaluate_result(tc, data)
            strategy = data.get("query_type_detected", "unknown") or "unknown"
            confidence = data.get("confidence_score", 0) or 0
            grade = data.get("grade", "F") or "F"

            # Check for routing errors: behavioral query misrouted
            if category == "behavioral" and strategy not in ("behavioral",):
                routing_errors += 1

        if passed:
            passed_count += 1
        else:
            failures.append({
                "query": query_text,
                "category": category,
                "expected_type": expected_type,
                "detected_type": strategy,
                "grade": grade,
                "reason": reason,
                "latency": round(latency, 3)
            })

        result_entry = {
            "query": query_text,
            "category": category,
            "expected_type": expected_type,
            "detected_type": strategy,
            "grade": grade,
            "confidence_score": confidence,
            "passed": passed,
            "reason": reason,
            "latency": round(latency, 3)
        }
        results.append(result_entry)

        # Submit feedback
        submit_feedback(query_text, strategy, passed, confidence)
        print(f"  {'PASS' if passed else 'FAIL'} | grade={grade} | strategy={strategy} | latency={latency:.2f}s")

    # Compute summary
    avg_latency = total_latency / total if total > 0 else 0
    pass_rate = passed_count / total if total > 0 else 0

    # Sort failures by latency descending for top 5
    failures_sorted = sorted(failures, key=lambda x: x["latency"], reverse=True)

    summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(pass_rate, 4),
        "avg_latency": round(avg_latency, 3),
        "routing_errors": routing_errors,
        "top_5_failures": failures_sorted[:5],
        "all_failures": failures_sorted
    }

    output = {
        "summary": summary,
        "results": results
    }

    # Write results
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Passed: {passed_count} ({pass_rate*100:.1f}%)")
    print(f"Failed: {total - passed_count}")
    print(f"Avg latency: {avg_latency:.3f}s")
    print(f"Routing errors (behavioral misrouted): {routing_errors}")
    print(f"\nTop 5 failures:")
    for j, f in enumerate(failures_sorted[:5]):
        print(f"  {j+1}. [{f['category']}] {f['query'][:50]}... | grade={f['grade']} | reason={f['reason']}")
    print(f"\nResults written to: {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    run_tests()
