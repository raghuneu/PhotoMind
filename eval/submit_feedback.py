#!/usr/bin/env python3
"""Submit RL feedback to production PhotoMind based on judge verdicts."""
import json, urllib.request, urllib.error, time

JUDGE = "eval/results/judge_scores.json"
RAW = "eval/results/browser_sweep_raw.json"
BASE = "https://srraghuram-photomind.hf.space"

scores = {j["id"]: j for j in json.load(open(JUDGE))}
raw = {r["id"]: r for r in json.load(open(RAW))}

submitted = 0
skipped = 0
strategy_wins = {}
for qid, j in scores.items():
    r = raw[qid]
    strategy = r.get("detected") or "semantic"
    label = j["feedback_label"]
    if label == "neutral":
        skipped += 1; continue
    was_correct = (label == "positive")
    payload = {
        "query": r["query"],
        "strategy": strategy,
        "was_correct": was_correct,
        "confidence_score": float(r.get("score") or 0.0),
    }
    req = urllib.request.Request(
        f"{BASE}/api/feedback",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            _ = resp.read()
        submitted += 1
        k = (strategy, was_correct)
        strategy_wins[k] = strategy_wins.get(k, 0) + 1
    except Exception as e:
        print(f"[{qid}] FAIL: {e}")
    time.sleep(0.05)

print(f"Submitted: {submitted}, skipped (neutral): {skipped}")
print("Per-strategy correct/incorrect:")
for (s, ok), n in sorted(strategy_wins.items()):
    print(f"  {s:12} {'correct' if ok else 'incorrect'}: {n}")

# Fetch updated stats
try:
    with urllib.request.urlopen(f"{BASE}/api/feedback/stats", timeout=15) as r:
        stats = json.loads(r.read())
    print("\nUpdated feedback stats:")
    print(json.dumps(stats, indent=2)[:1500])
except Exception as e:
    print(f"stats fetch failed: {e}")
