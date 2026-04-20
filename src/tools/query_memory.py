"""QueryMemoryStore — episodic memory for past query-outcome pairs.

Provides a lightweight, JSON-backed store that records every query the
system processes along with its routing decision, confidence grade, and
outcome.  This serves three purposes:

1. **Feedback loop for RL** — After each query, the bandit and DQN can
   receive delayed feedback (e.g., user corrections).  The memory store
   persists this so the next training run incorporates real-world signal.

2. **Duplicate detection** — If the user asks the exact same question
   again, the system can return the cached answer instantly (with a
   staleness warning if the knowledge base has been updated since).
   Uses exact string matching (see ``find_previous()``).

3. **Analytics** — The ``get_performance_summary()`` method aggregates
   routing accuracy and grade distributions for evaluation.

Note: This module is not currently integrated into the main query
pipeline. It is provided as infrastructure for future online learning.

Thread safety: file writes use a simple open-write-close pattern.  For
single-user desktop usage this is sufficient; a production deployment
would swap in SQLite or Redis.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QueryRecord:
    """Single query-outcome record."""
    query: str
    timestamp: float
    routing_source: str          # "rl_bandit" | "rule_based"
    strategy_used: str           # "factual" | "semantic" | "behavioral"
    confidence_grade: str        # A-F
    confidence_score: float
    source_photos: list = field(default_factory=list)
    user_feedback: Optional[str] = None   # "correct" | "incorrect" | None
    kb_version: Optional[str] = None      # hash of photo_index.json at query time


class QueryMemoryStore:
    """Persistent episodic memory for query history.

    Parameters
    ----------
    store_path : str
        Path to the JSON file backing the store.  Created on first write.
    max_records : int
        Maximum number of records to retain (FIFO eviction).
    """

    def __init__(
        self,
        store_path: str = "./knowledge_base/query_memory.json",
        max_records: int = 1000,
    ):
        self.store_path = store_path
        self.max_records = max_records
        self._records: list[dict] = []
        self._load()

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                self._records = data.get("records", [])
            except (json.JSONDecodeError, KeyError):
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump({"records": self._records}, f, indent=2)

    # ── Write ───────────────────────────────────────────────────────────

    def record_query(self, record: QueryRecord) -> None:
        """Append a query record, evicting oldest if over capacity."""
        self._records.append(asdict(record))
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
        self._save()

    def record_feedback(self, query: str, feedback: str) -> bool:
        """Attach user feedback to the most recent matching query.

        Returns True if a matching record was found and updated.
        """
        for record in reversed(self._records):
            if record["query"] == query:
                record["user_feedback"] = feedback
                self._save()
                return True
        return False

    # ── Read ────────────────────────────────────────────────────────────

    def find_previous(self, query: str) -> Optional[dict]:
        """Return the most recent record for an identical query, or None."""
        for record in reversed(self._records):
            if record["query"] == query:
                return record
        return None

    def get_performance_summary(self) -> dict:
        """Aggregate routing and confidence statistics.

        Returns
        -------
        dict with keys:
            total_queries, grade_distribution, routing_source_counts,
            strategy_counts, feedback_accuracy (if feedback available)
        """
        if not self._records:
            return {"total_queries": 0}

        grades = {}
        routing_sources = {}
        strategies = {}
        correct = 0
        has_feedback = 0

        for r in self._records:
            g = r.get("confidence_grade", "?")
            grades[g] = grades.get(g, 0) + 1

            rs = r.get("routing_source", "unknown")
            routing_sources[rs] = routing_sources.get(rs, 0) + 1

            s = r.get("strategy_used", "unknown")
            strategies[s] = strategies.get(s, 0) + 1

            fb = r.get("user_feedback")
            if fb is not None:
                has_feedback += 1
                if fb == "correct":
                    correct += 1

        summary = {
            "total_queries": len(self._records),
            "grade_distribution": grades,
            "routing_source_counts": routing_sources,
            "strategy_counts": strategies,
        }
        if has_feedback > 0:
            summary["feedback_accuracy"] = round(correct / has_feedback, 3)
            summary["feedback_count"] = has_feedback

        return summary

    @property
    def size(self) -> int:
        return len(self._records)
