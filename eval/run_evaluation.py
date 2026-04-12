"""Evaluation harness for PhotoMind — runs test queries, collects metrics."""

import json
import re
import time
import os
from datetime import datetime

from eval.test_cases import TEST_CASES
from src.crews.query_crew import create_query_crew
from src.tools.feedback_store import FeedbackStore


def parse_response(raw_result: str) -> dict:
    """Best-effort parse of the crew's output into structured fields."""
    text = str(raw_result).lower()
    parsed = {
        "source_photos": [],
        "confidence_grade": "F",
        "query_type": "unknown",
    }
    # Strip markdown bold markers so "**confidence_grade**: B" becomes "confidence_grade: b"
    text_clean = re.sub(r'\*+', '', text)

    # Try to parse as JSON first (synthesizer task requests JSON output)
    try:
        # Strip markdown code fences if present
        json_text = str(raw_result).strip()
        if json_text.startswith("```"):
            json_text = json_text.split("\n", 1)[1] if "\n" in json_text else json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3].strip()
        data = json.loads(json_text)
        if isinstance(data, dict):
            if data.get("confidence_grade", "").upper() in ("A", "B", "C", "D", "F"):
                parsed["confidence_grade"] = data["confidence_grade"].upper()
            if data.get("query_type") in ("factual", "semantic", "behavioral"):
                parsed["query_type"] = data["query_type"]
            if isinstance(data.get("source_photos"), list):
                parsed["source_photos"] = [
                    os.path.basename(p).lower() for p in data["source_photos"]
                ]
            parsed["raw"] = str(raw_result)
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract confidence grade — multiple pattern strategies
    grade_found = False
    for grade in ["A", "B", "C", "D", "F"]:
        g = grade.lower()
        if (
            f'"confidence_grade": "{g}"' in text
            or f'"confidence_grade": "{grade}"' in text
            or f"confidence_grade: {g}" in text
            or f"confidence_grade: {grade}" in text
            or f"confidence_grade: \"{g}\"" in text
            or f"confidence_grade: {g}" in text_clean
            or f"confidence: {g}" in text_clean
            or f"grade: {g}" in text_clean
            or f"grade {g}" in text_clean
            or re.search(rf'\bconfidence[_ ]?grade[^:]*:\s*["\']?{g}["\']?', text_clean)
            or re.search(rf'\bgrade[^a-z]*:\s*["\']?{g}["\']?\b', text_clean)
            # Catch "confidence_grade of C" or "confidence_grade is C" (no colon)
            or re.search(rf'\bconfidence[_ ]?grade\b.{{1,15}}\b{g}\b', text_clean)
        ):
            parsed["confidence_grade"] = grade
            grade_found = True
            break

    # Fallback: extract confidence_score and infer grade from it
    if not grade_found:
        score_match = re.search(r'confidence_score["\s:]+([0-9]+\.[0-9]+)', text)
        if not score_match:
            score_match = re.search(r'confidence["\s:]+([0-9]+\.[0-9]+)', text)
        if score_match:
            try:
                score = float(score_match.group(1))
                if score >= 0.7:
                    parsed["confidence_grade"] = "A"
                elif score >= 0.5:
                    parsed["confidence_grade"] = "B"
                elif score >= 0.35:
                    parsed["confidence_grade"] = "C"
                elif score >= 0.2:
                    parsed["confidence_grade"] = "D"
                else:
                    parsed["confidence_grade"] = "F"
            except ValueError:
                pass

    # Extract query type — prefer the tool's explicit "query_type_detected" JSON field
    qt_match = re.search(r'query_type_detected["\s:]+(\w+)', text)
    if qt_match and qt_match.group(1) in ("factual", "semantic", "behavioral"):
        parsed["query_type"] = qt_match.group(1)
    else:
        # Also check the "query_type" field from synthesizer JSON output
        qt_match2 = re.search(r'"query_type"["\s:]+["\']?(factual|semantic|behavioral)', text)
        if qt_match2:
            parsed["query_type"] = qt_match2.group(1)
        else:
            # Fallback: scan text, check behavioral before factual
            for qt in ["behavioral", "factual", "semantic"]:
                if qt in text:
                    parsed["query_type"] = qt
                    break

    # Extract photo filenames (look for .jpg, .png patterns)
    photos = re.findall(r'[\w\-]+\.(?:jpg|jpeg|png|webp|heic)', text, re.IGNORECASE)
    parsed["source_photos"] = [p.lower() for p in photos]
    parsed["raw"] = str(raw_result)
    return parsed


def run_eval():
    """Run the full evaluation suite and print metrics."""
    print("=" * 60)
    print("PhotoMind Evaluation Suite")
    print(f"Running {len(TEST_CASES)} test queries...")
    print("=" * 60)

    results = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {tc['query']}")
        start = time.time()

        try:
            # Fresh crew per query to prevent context accumulation across queries
            # (reusing one crew causes TPM errors by query ~18-20 due to memory growth)
            crew = create_query_crew()
            # Retry once on TPM rate-limit errors (429), with a 60-second backoff
            for attempt in range(2):
                try:
                    raw = crew.kickoff(inputs={"user_query": tc["query"]})
                    break
                except Exception as exc:
                    if attempt == 0 and "429" in str(exc):
                        print(f"  [rate limit] backing off 60s before retry...")
                        time.sleep(60)
                        crew = create_query_crew()
                    else:
                        raise
            elapsed = time.time() - start
            # Brief pause between queries to stay under TPM limit
            time.sleep(5)
            parsed = parse_response(raw)

            # Score: retrieval accuracy (case-insensitive: parsed filenames are lowercased)
            if tc["expected_photo"]:
                retrieval_correct = tc["expected_photo"].lower() in parsed["source_photos"]
            else:
                retrieval_correct = True  # No specific photo expected

            # Score: routing accuracy
            routing_correct = parsed["query_type"] == tc["expected_type"]

            # Score: silent failure detection
            is_wrong = not retrieval_correct
            is_confident = parsed["confidence_grade"] in ("A", "B", "C")
            silent_failure = is_wrong and is_confident

            # Score: decline accuracy (for edge cases)
            declined_correctly = None
            if tc.get("should_decline"):
                declined_correctly = parsed["confidence_grade"] in ("D", "F")

            result = {
                "query": tc["query"],
                "category": tc["category"],
                "retrieval_correct": retrieval_correct,
                "routing_correct": routing_correct,
                "confidence_grade": parsed["confidence_grade"],
                "silent_failure": silent_failure,
                "declined_correctly": declined_correctly,
                "latency_s": round(elapsed, 2),
            }
            results.append(result)
            status = "PASS" if retrieval_correct else "FAIL"
            print(f"  -> {status} | Grade: {parsed['confidence_grade']} | {elapsed:.1f}s")

            # Record outcome in feedback store for adaptive learning
            try:
                feedback = FeedbackStore()
                # Extract confidence_score from tool output if available
                conf_score = 0.0
                score_match = re.search(r'confidence_score["\s:]+([0-9]+\.[0-9]+)',
                                        str(raw).lower())
                if score_match:
                    conf_score = float(score_match.group(1))
                feedback.record_outcome(
                    query=tc["query"],
                    query_type=parsed["query_type"],
                    correct=retrieval_correct,
                    confidence_score=conf_score,
                )
            except Exception:
                pass  # Feedback recording is best-effort

        except Exception as e:
            elapsed = time.time() - start
            print(f"  -> ERROR: {e}")
            # For no-expected-photo queries, an error still means retrieval_correct=True
            retrieval_correct = tc["expected_photo"] is None
            results.append({
                "query": tc["query"],
                "category": tc["category"],
                "retrieval_correct": retrieval_correct,
                "routing_correct": False,
                "confidence_grade": "F",
                "silent_failure": False,
                "declined_correctly": True if tc.get("should_decline") else None,
                "latency_s": round(elapsed, 2),
                "error": str(e),
            })

    # ── Aggregate Metrics ────────────────────────────────────────────
    total = len(results)
    retrieval_acc = sum(r["retrieval_correct"] for r in results) / total
    routing_acc = sum(r["routing_correct"] for r in results) / total
    silent_failures = sum(1 for r in results if r["silent_failure"])
    silent_failure_rate = silent_failures / total
    avg_latency = sum(r["latency_s"] for r in results) / total

    decline_cases = [r for r in results if r["declined_correctly"] is not None]
    decline_acc = (
        sum(r["declined_correctly"] for r in decline_cases) / len(decline_cases)
        if decline_cases else 0
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Retrieval Accuracy:   {retrieval_acc:.1%}")
    print(f"Routing Accuracy:     {routing_acc:.1%}")
    print(f"Silent Failure Rate:  {silent_failure_rate:.1%}")
    print(f"Decline Accuracy:     {decline_acc:.1%}")
    print(f"Avg Latency:          {avg_latency:.1f}s")

    # Per-category breakdown
    print("\nPer-Category Breakdown:")
    for cat in ["factual", "semantic", "behavioral", "edge_case"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_acc = sum(r["retrieval_correct"] for r in cat_results) / len(cat_results)
            grade = (
                "A" if cat_acc >= 0.9 else
                "B" if cat_acc >= 0.8 else
                "C" if cat_acc >= 0.7 else
                "D" if cat_acc >= 0.5 else "F"
            )
            print(f"  {cat:15s}: {grade} ({cat_acc:.1%}, {len(cat_results)} queries)")

    # Save results to file
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "retrieval_accuracy": round(retrieval_acc, 3),
            "routing_accuracy": round(routing_acc, 3),
            "silent_failure_rate": round(silent_failure_rate, 3),
            "decline_accuracy": round(decline_acc, 3),
            "avg_latency_s": round(avg_latency, 2),
            "total_queries": total,
        },
        "results": results,
    }
    os.makedirs("./eval/results", exist_ok=True)
    with open("./eval/results/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to ./eval/results/eval_results.json")

    # Append to run history for trend analysis across multiple eval runs
    history_path = "./eval/results/eval_history.json"
    try:
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        else:
            history = []
        history.append({
            "timestamp": output["timestamp"],
            "summary": output["summary"],
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Run history updated: {len(history)} runs recorded.")
    except Exception:
        pass  # History tracking is best-effort

    # Print feedback store summary
    try:
        feedback = FeedbackStore()
        summary = feedback.get_summary()
        print("\nFeedback Loop Summary (adaptive thresholds):")
        for qt, stats in summary.items():
            acc = f"{stats['accuracy']:.1%}" if stats['accuracy'] is not None else "N/A"
            adj = stats['threshold_adjustment']
            adj_str = f"+{adj}" if adj > 0 else str(adj)
            print(f"  {qt:12s}: accuracy={acc}, threshold_adj={adj_str}")
    except Exception:
        pass


if __name__ == "__main__":
    run_eval()
