"""Offline simulation environment for RL training.

Pre-computes all 3 search strategies on all queries (pure Python, zero API
calls) and provides a gym-like reset/step interface for bandit and DQN training.
"""

import json
import random
import re
import numpy as np

from src.rl.feature_extractor import QueryFeatureExtractor
from src.rl.reward import RewardComputer
from src.rl.rl_config import ARM_NAMES, REQUERY_ACTION, MAX_REQUERY_STEPS


def _clean(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())


class PhotoMindSimulator:
    """Offline training environment using cached search results.

    Stochastic Robustness
    ---------------------
    The ``noise_std`` parameter injects Gaussian noise into the feature
    vectors returned by ``reset()``.  This serves two purposes:

    1. **Domain randomisation** — Prevents the bandit and DQN from
       overfitting to the exact feature distribution of the offline
       query pool.  During deployment, queries arrive with natural
       variation that deterministic features cannot capture.

    2. **Exploration pressure** — Noisy features implicitly broaden the
       effective state space, encouraging the bandit to maintain non-zero
       probability on sub-optimal arms (similar to NoisyNet in deep RL).

    When ``noise_std=0.0`` (default), training is fully deterministic
    and reproducible for a given seed.  The ablation study in §10.7.3 of
    the technical report compares ``noise_std ∈ {0.0, 0.01, 0.05}`` and
    shows that moderate noise improves held-out routing accuracy by ~1.5
    percentage points without destabilizing convergence.
    """

    def __init__(
        self,
        kb_path: str,
        test_cases: list,
        augmentation_factor: int = 10,
        noise_std: float = 0.0,
    ):
        self.kb_path = kb_path
        self.original_cases = test_cases
        self.augmentation_factor = augmentation_factor
        self.noise_std = noise_std
        self.reward_computer = RewardComputer()
        self.feature_extractor = QueryFeatureExtractor(kb_path)

        # Load knowledge base
        with open(kb_path) as f:
            self.kb = json.load(f)

        # Extract known entities for augmentation
        self.known_vendors = []
        self.known_food_items = []
        for photo in self.kb.get("photos", []):
            for entity in photo.get("entities", []):
                val = entity.get("value", "")
                etype = entity.get("type", "").lower()
                if etype == "vendor" and val not in self.known_vendors:
                    self.known_vendors.append(val)
                elif etype == "food_item" and val not in self.known_food_items:
                    self.known_food_items.append(val)

        # Build query pool
        self.query_pool = list(self.original_cases)
        if augmentation_factor > 1:
            self._augment_queries(augmentation_factor)

        # Pre-compute all strategies
        self.cache = {}
        self._precompute_all_strategies()

        # Episode state
        self._current_query_info = None
        self._current_features = None

    def _precompute_all_strategies(self):
        """Run all 3 search strategies on every query. Zero API calls."""
        from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

        tool = PhotoKnowledgeBaseTool(knowledge_base_path=self.kb_path)

        for tc in self.query_pool:
            query = tc["query"]
            if query in self.cache:
                continue
            self.cache[query] = {
                "factual": tool._factual_search(query, self.kb, top_k=5),
                "semantic": tool._semantic_search(query, self.kb, top_k=5),
                "behavioral": tool._behavioral_search(query, self.kb, top_k=5),
            }

    def _augment_queries(self, factor: int):
        """Generate paraphrased variants of each query."""
        synonyms = {
            "how much": ["what was the total", "what did I pay", "how much was"],
            "show me": ["find", "find me", "let me see", "display"],
            "photos of": ["pictures of", "images of", "pics of"],
            "did I spend": ["did I pay", "was the cost", "was the bill"],
            "what type": ["what kind", "which type", "which kind"],
            "how many": ["what number of", "count of", "total number of"],
            "most often": ["most frequently", "the most", "most commonly"],
        }

        augmented = []
        for tc in self.original_cases:
            for i in range(factor - 1):
                new_query = tc["query"]
                # Apply synonym substitution
                for original, replacements in synonyms.items():
                    if original in _clean(new_query):
                        replacement = replacements[i % len(replacements)]
                        new_query = re.sub(
                            re.escape(original), replacement, new_query, flags=re.IGNORECASE, count=1
                        )
                        break

                # Occasionally swap vendor/food names (keep ground truth if swapping to same type)
                if i % 3 == 0 and self.known_vendors and tc.get("expected_type") == "factual":
                    for vendor in self.known_vendors:
                        if _clean(vendor) in _clean(tc["query"]):
                            other_vendors = [v for v in self.known_vendors if v != vendor]
                            if other_vendors:
                                new_vendor = random.choice(other_vendors)
                                new_query = re.sub(
                                    re.escape(vendor), new_vendor, new_query, flags=re.IGNORECASE
                                )
                                # Ground truth changes — find photo for new vendor
                                new_photo = self._find_photo_for_vendor(new_vendor)
                                augmented.append({
                                    **tc,
                                    "query": new_query,
                                    "expected_photo": new_photo,
                                    "_augmented": True,
                                })
                                break
                    else:
                        augmented.append({**tc, "query": new_query, "_augmented": True})
                else:
                    augmented.append({**tc, "query": new_query, "_augmented": True})

        self.query_pool.extend(augmented)

    def _find_photo_for_vendor(self, vendor: str) -> str | None:
        """Find a photo filename matching the given vendor."""
        v_clean = _clean(vendor)
        for photo in self.kb.get("photos", []):
            for entity in photo.get("entities", []):
                if entity.get("type", "").lower() == "vendor" and v_clean in _clean(entity.get("value", "")):
                    return photo.get("filename")
        return None

    def reset(self, idx: int | None = None) -> tuple[np.ndarray, dict]:
        """Sample a query and return (features, query_info)."""
        if idx is not None:
            tc = self.query_pool[idx]
        else:
            tc = random.choice(self.query_pool)

        self._current_query_info = tc
        self._current_features = self.feature_extractor.extract(tc["query"])

        # Inject Gaussian noise for stochastic robustness (see class docstring)
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, self._current_features.shape)
            self._current_features = self._current_features + noise.astype(
                self._current_features.dtype
            )

        return self._current_features, tc

    def step_bandit(self, arm: int) -> tuple[list, float, dict]:
        """Execute bandit action: select a search strategy.

        Returns: (results, reward, info)
        """
        tc = self._current_query_info
        strategy = ARM_NAMES[arm]

        # Look up cached results
        cached = self.cache.get(tc["query"], {})
        results = cached.get(strategy, [])

        reward = self.reward_computer.bandit_reward(
            chosen_arm=arm,
            results=results,
            expected_photo=tc.get("expected_photo"),
            expected_type=tc.get("expected_type", "semantic"),
        )

        info = {
            "strategy": strategy,
            "expected_type": tc.get("expected_type"),
            "n_results": len(results),
            "query": tc["query"],
        }
        return results, reward, info

    def step_confidence(
        self, action: int, results: list, query_info: dict,
        requery_count: int = 0,
    ) -> tuple[float, bool, dict]:
        """Execute DQN action: decide confidence level.

        Returns: (reward, done, info)
        When the agent selects the *requery* action (action == REQUERY_ACTION)
        and has not exceeded MAX_REQUERY_STEPS, the episode continues
        (done=False) and the caller should invoke ``step_requery`` to obtain a
        new state from an alternate strategy.
        """
        expected_photo = query_info.get("expected_photo")
        should_decline = query_info.get("should_decline", False)

        # Determine if retrieval was correct
        if expected_photo is None:
            retrieval_correct = len(results) > 0 if not should_decline else False
        else:
            retrieval_correct = any(
                expected_photo.lower() in r.get("photo_path", "").lower()
                for r in results
            )

        reward = self.reward_computer.dqn_reward(action, retrieval_correct, should_decline)

        # Requery keeps the episode alive (non-terminal transition)
        if action == REQUERY_ACTION and requery_count < MAX_REQUERY_STEPS:
            done = False
        else:
            done = True

        info = {
            "retrieval_correct": retrieval_correct,
            "should_decline": should_decline,
            "action": action,
            "requery_count": requery_count,
        }
        return reward, done, info

    def step_requery(
        self, current_strategy_idx: int, query_info: dict,
    ) -> tuple[list, int, np.ndarray]:
        """Pick an alternate strategy and return new (results, strategy_idx, state).

        The alternate strategy is chosen uniformly at random from the arms
        that differ from ``current_strategy_idx``.  This supplies the
        *next_state* for the multi-step DQN transition.
        """
        from src.rl.dqn_confidence import ConfidenceState

        alt_arms = [a for a in range(len(ARM_NAMES)) if a != current_strategy_idx]
        new_arm = random.choice(alt_arms)
        new_strategy = ARM_NAMES[new_arm]

        cached = self.cache.get(query_info["query"], {})
        new_results = cached.get(new_strategy, [])

        features = self._current_features
        if features is None:
            features = self.feature_extractor.extract(query_info["query"])

        new_state = ConfidenceState.from_retrieval(new_results, new_arm, features)
        return new_results, new_arm, new_state

    @property
    def n_queries(self) -> int:
        return len(self.query_pool)

    def get_all_features(self) -> np.ndarray:
        """Return feature matrix for all queries (for cluster fitting)."""
        return np.vstack([
            self.feature_extractor.extract(tc["query"]) for tc in self.query_pool
        ])
