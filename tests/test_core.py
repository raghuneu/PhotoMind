"""Unit tests for PhotoMind core components.

Tests reward computation, feature extraction, statistical analysis,
confidence state construction, DQN action-to-grade mapping, and
the train/test split integrity.
"""

import numpy as np
import pytest

from src.rl.rl_config import ARM_NAMES, REWARD_MATRIX, REQUERY_ACTION, DECLINE_ACTION
from src.rl.reward import RewardComputer
from src.rl.feature_extractor import QueryFeatureExtractor, _clean
from src.rl.dqn_confidence import ConfidenceState, action_to_grade
from eval.statistical_analysis import (
    confidence_interval, paired_t_test, cohens_d, format_ci,
)


# ── Reward computation ──────────────────────────────────────────────────────

class TestBanditReward:
    """Tests for RewardComputer.bandit_reward."""

    def test_correct_strategy_and_photo_found(self):
        results = [{"photo_path": "photos/receipt_001.jpg", "relevance_score": 0.9}]
        reward = RewardComputer.bandit_reward(
            chosen_arm=0, results=results,
            expected_photo="receipt_001", expected_type="factual",
        )
        assert reward == 1.0

    def test_photo_found_wrong_strategy(self):
        results = [{"photo_path": "photos/receipt_001.jpg", "relevance_score": 0.9}]
        reward = RewardComputer.bandit_reward(
            chosen_arm=1, results=results,  # semantic instead of factual
            expected_photo="receipt_001", expected_type="factual",
        )
        assert reward == 0.5

    def test_photo_not_found(self):
        results = [{"photo_path": "photos/other.jpg", "relevance_score": 0.3}]
        reward = RewardComputer.bandit_reward(
            chosen_arm=0, results=results,
            expected_photo="receipt_001", expected_type="factual",
        )
        assert reward == 0.0

    def test_aggregate_query_correct_strategy(self):
        """Aggregate queries (no expected_photo) should get 1.0 for correct strategy."""
        reward = RewardComputer.bandit_reward(
            chosen_arm=2, results=[],
            expected_photo=None, expected_type="behavioral",
        )
        assert reward == 1.0

    def test_aggregate_query_wrong_strategy(self):
        reward = RewardComputer.bandit_reward(
            chosen_arm=0, results=[],
            expected_photo=None, expected_type="behavioral",
        )
        assert reward == 0.3

    def test_empty_results_with_expected_photo(self):
        reward = RewardComputer.bandit_reward(
            chosen_arm=0, results=[],
            expected_photo="receipt_001", expected_type="factual",
        )
        assert reward == 0.0


class TestDQNReward:
    """Tests for RewardComputer.dqn_reward — verifies the reward matrix."""

    def test_silent_failure_penalized(self):
        """High-confidence on wrong retrieval should be -1.0."""
        reward = RewardComputer.dqn_reward(action=0, retrieval_correct=False, should_decline=False)
        assert reward == -1.0

    def test_correct_accept_high(self):
        reward = RewardComputer.dqn_reward(action=0, retrieval_correct=True, should_decline=False)
        assert reward == 1.0

    def test_correct_decline(self):
        reward = RewardComputer.dqn_reward(action=DECLINE_ACTION, retrieval_correct=True, should_decline=True)
        assert reward == 1.0

    def test_unnecessary_decline(self):
        """Declining a correct retrieval that shouldn't be declined costs -0.3."""
        reward = RewardComputer.dqn_reward(action=DECLINE_ACTION, retrieval_correct=True, should_decline=False)
        assert reward == -0.3

    def test_requery_step_cost(self):
        """Requery action should have a small step cost of -0.1."""
        reward = RewardComputer.dqn_reward(action=REQUERY_ACTION, retrieval_correct=True, should_decline=False)
        assert reward == -0.1
        reward = RewardComputer.dqn_reward(action=REQUERY_ACTION, retrieval_correct=False, should_decline=False)
        assert reward == -0.1

    def test_unknown_key_returns_zero(self):
        reward = RewardComputer.dqn_reward(action=99, retrieval_correct=True, should_decline=False)
        assert reward == 0.0

    def test_all_matrix_keys_covered(self):
        """Every (action, correct, decline) combo for valid actions should exist."""
        for action in range(5):  # 0..4: accept_high, accept_moderate, hedge, requery, decline
            for correct in [True, False]:
                for decline in [True, False]:
                    assert (action, correct, decline) in REWARD_MATRIX


# ── Feature extraction ──────────────────────────────────────────────────────

class TestFeatureExtractor:
    """Tests for QueryFeatureExtractor.extract."""

    @pytest.fixture
    def extractor(self):
        return QueryFeatureExtractor(kb_path="./knowledge_base/photo_index.json")

    def test_output_shape(self, extractor):
        features = extractor.extract("How much did I spend at Trader Joe's?")
        assert features.shape == (12,)
        assert features.dtype == np.float32

    def test_amount_keyword_detected(self, extractor):
        features = extractor.extract("How much did I spend?")
        assert features[2] == 1.0  # has_amount_keyword

    def test_behavioral_keyword_detected(self, extractor):
        features = extractor.extract("Which store do I shop at most often?")
        assert features[5] == 1.0  # has_behavioral_keyword

    def test_semantic_keyword_detected(self, extractor):
        features = extractor.extract("Show me photos of food")
        assert features[6] == 1.0  # has_semantic_keyword

    def test_wh_question_type(self, extractor):
        features = extractor.extract("What did I buy at Target?")
        assert features[8] == 1.0  # wh-question

    def test_imperative_type(self, extractor):
        features = extractor.extract("Show me receipts")
        assert features[10] == 1.0  # imperative

    def test_negation_detected(self, extractor):
        features = extractor.extract("Show me photos not from Boston")
        assert features[7] == 1.0

    def test_no_negation(self, extractor):
        features = extractor.extract("Show me photos from Boston")
        assert features[7] == 0.0

    def test_clean_strips_punctuation(self):
        assert _clean("Hello, World!") == "hello world"
        assert _clean("$12.99") == "1299"


# ── Confidence state ────────────────────────────────────────────────────────

class TestConfidenceState:
    """Tests for ConfidenceState.from_retrieval."""

    def test_empty_results(self):
        features = np.zeros(12, dtype=np.float32)
        state = ConfidenceState.from_retrieval([], strategy_idx=0, query_features=features)
        assert state.shape == (8,)
        assert state[0] == 0.0  # top score
        assert state[2] == 0.0  # num results

    def test_single_result(self):
        results = [{"relevance_score": 0.8, "evidence": "entity match", "photo_path": "x.jpg"}]
        features = np.array([0.5] + [0.0] * 11, dtype=np.float32)
        state = ConfidenceState.from_retrieval(results, strategy_idx=1, query_features=features)
        assert state[0] == pytest.approx(0.8)   # top score
        assert state[1] == pytest.approx(0.0)   # gap (only 1 result)
        assert state[2] == pytest.approx(0.1)   # 1/10
        assert state[3] == pytest.approx(0.5)   # strategy_idx 1 / 2.0
        assert state[4] == pytest.approx(0.5)   # query_features[0]
        assert state[5] == 1.0                   # entity in evidence

    def test_strategy_normalization(self):
        features = np.zeros(12, dtype=np.float32)
        for idx in range(3):
            state = ConfidenceState.from_retrieval([], strategy_idx=idx, query_features=features)
            assert state[3] == pytest.approx(idx / 2.0)


# ── Action-to-grade mapping ────────────────────────────────────────────────

class TestActionToGrade:

    def test_accept_high_with_high_score(self):
        assert action_to_grade(0, score=0.9) == "A"

    def test_accept_high_with_low_score(self):
        assert action_to_grade(0, score=0.3) == "B"

    def test_accept_moderate(self):
        assert action_to_grade(1) == "C"

    def test_hedge(self):
        assert action_to_grade(2) == "D"

    def test_requery(self):
        assert action_to_grade(REQUERY_ACTION) == "REQUERY"

    def test_decline(self):
        assert action_to_grade(DECLINE_ACTION) == "F"


# ── Statistical analysis ───────────────────────────────────────────────────

class TestStatisticalAnalysis:

    def test_ci_single_value(self):
        mean, lower, upper, margin = confidence_interval([0.5])
        assert mean == 0.5
        assert margin == 0.0

    def test_ci_multiple_values(self):
        mean, lower, upper, margin = confidence_interval([0.8, 0.8, 0.8, 0.8, 0.8])
        assert mean == pytest.approx(0.8)
        assert margin == pytest.approx(0.0)

    def test_ci_with_variance(self):
        data = [0.7, 0.8, 0.9, 0.85, 0.75]
        mean, lower, upper, margin = confidence_interval(data)
        assert lower < mean < upper
        assert margin > 0

    def test_paired_t_test_identical(self):
        t_stat, p_val = paired_t_test([0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
        assert t_stat == 0.0
        assert p_val == 1.0

    def test_paired_t_test_constant_difference(self):
        """Constant non-zero difference should give p near 0."""
        t_stat, p_val = paired_t_test([0.5, 0.5, 0.5, 0.5, 0.5],
                                       [0.8, 0.8, 0.8, 0.8, 0.8])
        assert p_val < 0.001

    def test_cohens_d_no_difference(self):
        d = cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert d == 0.0

    def test_cohens_d_constant_difference(self):
        d = cohens_d([0.5, 0.5, 0.5], [0.8, 0.8, 0.8])
        assert d == float('inf')

    def test_cohens_d_with_variance(self):
        d = cohens_d([0.5, 0.6, 0.7], [0.8, 0.9, 1.0])
        assert d > 0  # positive effect

    def test_format_ci_percentage(self):
        result = format_ci(0.875, 0.85, 0.90)
        assert "87.5%" in result
        assert "[" in result

    def test_format_ci_clamp(self):
        """Values exceeding [0, 1] should be clamped."""
        result = format_ci(0.95, -0.02, 1.05)
        assert "0.0%" in result or "[0.0%" in result
        assert "100.0%" in result


# ── Train/test split integrity ──────────────────────────────────────────────

class TestTrainTestSplit:

    def test_split_sizes(self):
        from eval.expanded_test_cases import (
            ALL_TEST_CASES, TRAIN_TEST_CASES, HELD_OUT_TEST_CASES,
        )
        assert len(ALL_TEST_CASES) == 56
        assert len(TRAIN_TEST_CASES) == 42
        assert len(HELD_OUT_TEST_CASES) == 14

    def test_no_overlap(self):
        from eval.expanded_test_cases import TRAIN_TEST_CASES, HELD_OUT_TEST_CASES
        train_queries = {tc["query"] for tc in TRAIN_TEST_CASES}
        held_out_queries = {tc["query"] for tc in HELD_OUT_TEST_CASES}
        assert train_queries.isdisjoint(held_out_queries)

    def test_union_equals_all(self):
        from eval.expanded_test_cases import (
            ALL_TEST_CASES, TRAIN_TEST_CASES, HELD_OUT_TEST_CASES,
        )
        all_queries = {tc["query"] for tc in ALL_TEST_CASES}
        combined = {tc["query"] for tc in TRAIN_TEST_CASES} | {tc["query"] for tc in HELD_OUT_TEST_CASES}
        assert all_queries == combined

    def test_held_out_category_coverage(self):
        """Every category should be represented in the held-out set."""
        from eval.expanded_test_cases import HELD_OUT_TEST_CASES
        categories = {tc["category"] for tc in HELD_OUT_TEST_CASES}
        assert categories == {"factual", "semantic", "behavioral", "edge_case", "ambiguous"}
