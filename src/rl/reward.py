"""Reward computation for bandit and DQN components."""

from src.rl.rl_config import ARM_NAMES, REWARD_MATRIX


class RewardComputer:
    """Computes rewards for both RL components."""

    @staticmethod
    def bandit_reward(
        chosen_arm: int,
        results: list,
        expected_photo: str | None,
        expected_type: str,
    ) -> float:
        """Reward for the contextual bandit's strategy selection.

        Returns:
            +1.0 — correct strategy AND expected photo found in results
            +0.5 — expected photo found but strategy didn't match expected_type
             0.0 — expected photo not found (or no expected photo for aggregate queries)
        """
        chosen_strategy = ARM_NAMES[chosen_arm]

        # For aggregate queries with no specific expected photo
        if expected_photo is None:
            return 1.0 if chosen_strategy == expected_type else 0.3

        # Check if expected photo appears in results
        photo_found = any(
            expected_photo.lower() in r.get("photo_path", "").lower()
            for r in results
        )

        if photo_found and chosen_strategy == expected_type:
            return 1.0
        elif photo_found:
            return 0.5
        else:
            return 0.0

    @staticmethod
    def dqn_reward(
        action: int,
        retrieval_correct: bool,
        should_decline: bool,
    ) -> float:
        """Reward for the DQN confidence calibrator's accept/hedge/decline decision."""
        key = (action, retrieval_correct, should_decline)
        return REWARD_MATRIX.get(key, 0.0)
