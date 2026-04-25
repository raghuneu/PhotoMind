"""Extract feature vectors from natural language queries for RL components.

Returns a 396-dim hybrid vector: 12 handcrafted lexical/structural features
(indices 0-11) concatenated with a 384-dim MiniLM sentence embedding
(indices 12-395).

The MiniLM encoder (sentence-transformers/all-MiniLM-L6-v2) is the same
model used for photo embeddings in the Qdrant index, so query features live
in the same semantic space as indexed photos. This lets KMeans cluster the
query space by actual meaning, not only by keyword patterns — a real
improvement over the prior 12-dim keyword-only features that collapsed
paraphrases into different clusters.

The handcrafted 12 are kept at the FRONT of the vector so that
ConfidenceState.from_retrieval (which reads query_features[0] as
"query length normalized" for DQN state dim 4) remains semantically valid
without any caller changes.
"""

import json
import re
import numpy as np

# MiniLM encoder is loaded lazily and cached at module scope so the 90 MB
# model doesn't reload on every QueryFeatureExtractor instantiation (used
# per-query inside the retrieval tool).
_ENCODER = None
_EMBED_DIM = 384
HANDCRAFTED_DIM = 12
FEATURE_DIM = HANDCRAFTED_DIM + _EMBED_DIM  # 396


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ENCODER


def _clean(text: str) -> str:
    """Lowercase and strip punctuation (matches photo_knowledge_base.py)."""
    return re.sub(r'[^\w\s]', '', text.lower())


class QueryFeatureExtractor:
    """Converts a natural language query into a 396-dim feature vector.

    Layout:
        [0:12]   handcrafted lexical/structural features (length, keyword flags)
        [12:396] MiniLM-L6-v2 sentence embedding (normalized by model)
    """

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

    def _handcrafted(self, query: str) -> np.ndarray:
        """Return the 12-dim handcrafted feature slice."""
        q = _clean(query)
        words = q.split()

        features = np.zeros(HANDCRAFTED_DIM, dtype=np.float32)

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

    def extract(self, query: str) -> np.ndarray:
        """Return a 396-dim hybrid feature vector (12 handcrafted + 384 MiniLM)."""
        handcrafted = self._handcrafted(query)
        try:
            embedding = _get_encoder().encode(
                query, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        except Exception:
            # If encoder fails to load (offline / no torch), fall back to zeros
            # so the bandit still runs on the handcrafted prefix.
            embedding = np.zeros(_EMBED_DIM, dtype=np.float32)
        return np.concatenate([handcrafted, embedding])

