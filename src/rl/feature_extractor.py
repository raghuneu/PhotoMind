"""Extract feature vectors from natural language queries for RL components."""

import json
import re
import numpy as np


def _clean(text: str) -> str:
    """Lowercase and strip punctuation (matches photo_knowledge_base.py)."""
    return re.sub(r'[^\w\s]', '', text.lower())


class QueryFeatureExtractor:
    """Converts a natural language query into a fixed-size feature vector."""

    def __init__(self, kb_path: str = "./knowledge_base/photo_index.json"):
        self.known_vendors = set()
        self.known_food_items = set()
        try:
            with open(kb_path) as f:
                kb = json.load(f)
            for photo in kb.get("photos", []):
                for entity in photo.get("entities", []):
                    val = _clean(entity.get("value", ""))
                    etype = entity.get("type", "").lower()
                    if etype == "vendor":
                        self.known_vendors.add(val)
                    elif etype == "food_item":
                        self.known_food_items.add(val)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def extract(self, query: str) -> np.ndarray:
        """Return a 12-dimensional feature vector for the given query."""
        q = _clean(query)
        words = q.split()

        features = np.zeros(12, dtype=np.float32)

        # 0: query length normalized
        features[0] = min(len(query) / 100.0, 1.0)
        # 1: word count normalized
        features[1] = min(len(words) / 20.0, 1.0)
        # 2: has amount keyword
        amount_kws = ["how much", "price", "cost", "total", "spend", "spent", "amount", "paid"]
        features[2] = 1.0 if any(kw in q for kw in amount_kws) else 0.0
        # 3: has date keyword
        date_kws = ["when", "date", "time", "day", "month", "year"]
        features[3] = 1.0 if any(kw in q for kw in date_kws) else 0.0
        # 4: has vendor keyword
        vendor_kws = ["receipt", "bill", "store", "shop", "vendor", "invoice", "payment"]
        features[4] = 1.0 if any(kw in q for kw in vendor_kws) else 0.0
        # 5: has behavioral keyword
        behavioral_kws = ["most", "often", "pattern", "how many", "favorite",
                          "frequently", "trend", "habit", "compare", "average"]
        features[5] = 1.0 if any(kw in q for kw in behavioral_kws) else 0.0
        # 6: has semantic keyword
        semantic_kws = ["show me", "look like", "photos of", "pictures of",
                        "find photos", "find a", "feel like", "scenic"]
        features[6] = 1.0 if any(kw in q for kw in semantic_kws) else 0.0
        # 7: has negation
        features[7] = 1.0 if any(w in words for w in ["not", "never", "no", "none"]) else 0.0
        # 8-10: question type (one-hot: wh-question, yes/no, imperative)
        if any(q.startswith(w) for w in ["what", "where", "when", "which", "how", "who", "why"]):
            features[8] = 1.0
        elif any(q.startswith(w) for w in ["is", "are", "do", "does", "did", "can", "could"]):
            features[9] = 1.0
        elif any(q.startswith(w) for w in ["show", "find", "get", "list", "tell"]):
            features[10] = 1.0
        # 11: known vendor match
        features[11] = 1.0 if any(v in q for v in self.known_vendors if len(v) > 2) else 0.0

        return features
