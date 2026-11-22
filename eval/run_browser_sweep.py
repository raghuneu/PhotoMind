#!/usr/bin/env python3
"""Drive 75-query sweep against production PhotoMind API.
Same endpoint the web UI calls (verified via browser fetch).
"""
import json, time, urllib.request, urllib.error, sys

MANIFEST = "eval/results/query_manifest_rand75.json"
OUT = "eval/results/browser_sweep_raw.json"
BASE = "https://srraghuram-photomind.hf.space"

manifest = json.load(open(MANIFEST))
results = []
for i, item in enumerate(manifest, 1):
    t0 = time.time()
    payload = json.dumps({"query": item["query"], "mode": "fast"}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/query", data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code; body = {"error": e.read().decode()[:200]}
    except Exception as e:
        status = 0; body = {"error": str(e)}
    ms = int((time.time() - t0) * 1000)
    results.append({
        "id": item["id"],
        "query": item["query"],
        "intent": item["intent"],
        "expected_photo": item.get("source_photo"),
        "should_decline": item.get("should_decline", False),
        "status": status, "ms": ms,
        "grade": body.get("confidence_grade"),
        "score": body.get("confidence_score"),
        "detected": body.get("query_type_detected"),
        "routing": body.get("routing_source"),
        "n": len(body.get("source_photos") or []),
        "source_photos": body.get("source_photos") or [],
        "answer": body.get("answer_summary"),
        "warning": body.get("warning"),
    })
    print(f"[{i:2}/75] {item['query'][:50]:<50} -> grade={body.get('confidence_grade')} n={len(body.get('source_photos') or [])} {ms}ms", flush=True)

json.dump(results, open(OUT, "w"), indent=2)
grades = {}
for r in results: grades[r["grade"] or "ERR"] = grades.get(r["grade"] or "ERR", 0) + 1
print(f"\nDONE. Wrote {OUT}. Grades: {grades}")
