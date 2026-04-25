"""Reward computation for bandit and DQN components."""

from src.rl.rl_config import ARM_NAMES, REWARD_MATRIX


def _result_text(r: dict) -> str:
    """Concatenate text fields from a search result for substring matching."""
    parts = [
        str(r.get("photo_path", "")),
        str(r.get("evidence", "")),
        str(r.get("description", "")),
        str(r.get("ocr_text", "")),
    ]
    # Flatten any entity list
    for ent in r.get("entities", []) or []:
        if isinstance(ent, dict):
            parts.append(str(ent.get("value", "")))
        else:
            parts.append(str(ent))
    return " ".join(parts).lower()


class RewardComputer:
    """Computes rewards for both RL components."""

    @staticmethod
    def bandit_reward(
        chosen_arm: int,
        results: list,
        expected_photo: str | None,
        expected_type: str,
        expected_top_entity: str | None = None,
    ) -> float:
        """Reward for the contextual bandit's strategy selection.

        Returns:
            +1.0 — correct strategy AND expected photo found in results
            +0.5 — expected photo found but strategy didn't match expected_type
             0.0 — expected photo not found (or no expected photo for aggregate queries)

        Aggregate queries (``expected_photo is None``) previously granted a
        flat +1.0/+0.3 purely on strategy-label match, regardless of whether
        any useful evidence was actually retrieved. This rewarded a policy
        that picked "behavioral" every time a test case was labeled behavioral
        even if the strategy returned zero results.

        When ``expected_top_entity`` is provided, we grade the aggregate
        outcome on substring evidence in the retrieved results (vendor, item,
        filename) instead. If no ``expected_top_entity`` is provided, we fall
        back to the legacy generous behavior for backward compatibility with
        older test cases.
        """
        chosen_strategy = ARM_NAMES[chosen_arm]

        # Aggregate queries (no single expected photo)
        if expected_photo is None:
            if expected_top_entity is not None:
                # Outcome-based: check if the expected entity appears anywhere
                # in the top-k results' evidence, description, or entities.
                needle = expected_top_entity.lower()
                found = any(needle in _result_text(r) for r in results)
                if found and chosen_strategy == expected_type:
                    return 1.0
                if found:
                    return 0.6
                # Strategy matched label but produced no useful evidence —
                # small credit to encourage correct routing without rewarding
                # empty retrieval.
                if chosen_strategy == expected_type:
                    return 0.2
                return 0.0
            # Legacy behavior for cases without expected_top_entity
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
