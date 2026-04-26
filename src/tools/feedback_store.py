"""
FeedbackStore — persistent feedback loop for PhotoMind.

Stores query outcomes and uses them to adjust confidence thresholds
and strategy routing weights. Implements the assignment's requirement
for "feedback loops for agent improvement."

After each evaluation run, per-strategy accuracy rates are computed.
If a strategy's accuracy drops below 70%, its confidence threshold is
boosted by +0.05 (more conservative — fewer false positives). If
accuracy is >= 90%, the threshold is lowered by -0.05 (less conservative).

Storage: knowledge_base/feedback_store.json
"""

import json
import os
from datetime import datetime, timezone


FEEDBACK_PATH = "./knowledge_base/feedback_store.json"


class FeedbackStore:
    """Persistent store for query outcomes and adaptive threshold adjustments."""

    def __init__(self, path: str = FEEDBACK_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        default = {
            "history": [],
            "strategy_stats": {
                "factual": {"correct": 0, "total": 0},
                "semantic": {"correct": 0, "total": 0},
                "behavioral": {"correct": 0, "total": 0},
                "embedding": {"correct": 0, "total": 0},
            },
            "confidence_adjustments": {
                "factual": 0.0,
                "semantic": 0.0,
                "behavioral": 0.0,
                "embedding": 0.0,
            },
        }
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "history" in data:
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return default

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _update_stats(self, query_type: str, correct: bool, n_results: int | None = None):
        """Update per-strategy accuracy stats and confidence adjustments.

        ``n_results`` lets the loop distinguish "missed retrieval" (no
        results, harmful to raise the threshold further) from "wrong
        retrieval" (results returned but judged incorrect — the classic
        false-positive case the threshold should guard against).
        """
        if query_type not in self.data["strategy_stats"]:
            self.data["strategy_stats"][query_type] = {"correct": 0, "total": 0}
        stats = self.data["strategy_stats"][query_type]
        stats["total"] += 1
        if correct:
            stats["correct"] += 1
        # P2: track missed-retrieval events separately so we don't raise
        # the threshold (which worsens miss rate) when the real problem
        # is retrieval recall, not calibration.
        if not correct and n_results is not None and n_results == 0:
            stats["missed"] = stats.get("missed", 0) + 1

        # Recompute adaptive confidence adjustment for this strategy.
        # P2: damp the swing to +/-0.03 and cap cumulative drift so a
        # bad run can't permanently push the threshold above 0.5.
        if stats["total"] >= 3:
            accuracy = stats["correct"] / stats["total"]
            miss_rate = stats.get("missed", 0) / stats["total"]
            current = self.data["confidence_adjustments"].get(query_type, 0.0)
            if accuracy < 0.7 and miss_rate < 0.5:
                # Only tighten when failures are false-positives, not misses.
                new_adj = min(current + 0.03, 0.10)
            elif accuracy >= 0.9:
                new_adj = max(current - 0.03, -0.10)
            elif miss_rate >= 0.5:
                # Predominantly missed retrievals — loosen to improve recall.
                new_adj = max(current - 0.03, -0.10)
            else:
                new_adj = current  # avoid churn in the 70–90% band
            self.data["confidence_adjustments"][query_type] = round(new_adj, 3)

    def record_outcome(
        self,
        query: str,
        query_type: str,
        correct: bool,
        confidence_score: float,
        n_results: int | None = None,
    ):
        """Record a query outcome for feedback learning."""
        self.data["history"].append({
            "query": query,
            "query_type": query_type,
            "correct": correct,
            "confidence_score": confidence_score,
            "n_results": n_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self._update_stats(query_type, correct, n_results)
        self._save()

    def get_confidence_adjustment(self, query_type: str) -> float:
        """Get the adaptive confidence threshold adjustment for a strategy.

        Returns a float to add to the base confidence_threshold:
          +0.05 if strategy accuracy < 70% (be more conservative)
          -0.05 if strategy accuracy >= 90% (be less conservative)
           0.0  otherwise
        """
        return self.data.get("confidence_adjustments", {}).get(query_type, 0.0)

    def get_strategy_accuracy(self, query_type: str) -> float | None:
        """Get the observed accuracy rate for a strategy, or None if < 3 samples."""
        stats = self.data.get("strategy_stats", {}).get(query_type)
        if stats and stats["total"] >= 3:
            return stats["correct"] / stats["total"]
        return None

    def record_rl_outcome(
        self,
        query: str,
        query_type: str,
        correct: bool,
        confidence_score: float,
        bandit_arm: int | None = None,
        dqn_action: int | None = None,
        bandit_reward: float | None = None,
        dqn_reward: float | None = None,
        n_results: int | None = None,
    ):
        """Record a query outcome with RL-specific fields."""
        self.data["history"].append({
            "query": query,
            "query_type": query_type,
            "correct": correct,
            "confidence_score": confidence_score,
            "bandit_arm": bandit_arm,
            "dqn_action": dqn_action,
            "bandit_reward": bandit_reward,
            "dqn_reward": dqn_reward,
            "n_results": n_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self._update_stats(query_type, correct, n_results)
        self._save()

    def get_summary(self) -> dict:
        """Return a summary of feedback data for agent context."""
        summary = {}
        for qt in ("factual", "semantic", "behavioral"):
            stats = self.data.get("strategy_stats", {}).get(qt, {})
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            adj = self.data.get("confidence_adjustments", {}).get(qt, 0.0)
            summary[qt] = {
                "queries_seen": total,
                "accuracy": round(correct / total, 3) if total > 0 else None,
                "threshold_adjustment": adj,
            }
        return summary
