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
            },
            "confidence_adjustments": {
                "factual": 0.0,
                "semantic": 0.0,
                "behavioral": 0.0,
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

    def _update_stats(self, query_type: str, correct: bool):
        """Update per-strategy accuracy stats and confidence adjustments."""
        if query_type not in self.data["strategy_stats"]:
            self.data["strategy_stats"][query_type] = {"correct": 0, "total": 0}
        stats = self.data["strategy_stats"][query_type]
        stats["total"] += 1
        if correct:
            stats["correct"] += 1

        # Recompute adaptive confidence adjustment for this strategy
        if stats["total"] >= 3:
            accuracy = stats["correct"] / stats["total"]
            if accuracy < 0.7:
                self.data["confidence_adjustments"][query_type] = 0.05
            elif accuracy >= 0.9:
                self.data["confidence_adjustments"][query_type] = -0.05
            else:
                self.data["confidence_adjustments"][query_type] = 0.0

    def record_outcome(
        self,
        query: str,
        query_type: str,
        correct: bool,
        confidence_score: float,
    ):
        """Record a query outcome for feedback learning."""
        self.data["history"].append({
            "query": query,
            "query_type": query_type,
            "correct": correct,
            "confidence_score": confidence_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self._update_stats(query_type, correct)
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self._update_stats(query_type, correct)
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
