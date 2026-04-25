"""Before/after comparison for demo video — shows rule-based vs RL routing."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.test_cases import TEST_CASES
from src.rl.simulation_env import PhotoMindSimulator
from src.rl.contextual_bandit import load_trained_bandit
from src.rl.dqn_confidence import load_trained_dqn, ConfidenceState, action_to_grade
from src.rl.rl_config import ARM_NAMES, ACTION_NAMES, BANDIT_MODEL_PATH, DQN_MODEL_PATH
from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool


def main():
    kb_path = "./knowledge_base/photo_index.json"
    sim = PhotoMindSimulator(kb_path, TEST_CASES, augmentation_factor=1)
    tool = PhotoKnowledgeBaseTool(knowledge_base_path=kb_path)

    bandit = load_trained_bandit(BANDIT_MODEL_PATH)
    dqn = load_trained_dqn(DQN_MODEL_PATH)

    demo_queries = [
        "How much did I spend at ALDI?",
        "Show me photos of pizza",
        "Which store do I shop at most often?",
        "What was my electric bill this month?",
        "Show me something I spent a lot on",
    ]

    print("=" * 70)
    print("PhotoMind: Rule-Based vs RL-Powered Query Processing")
    print("=" * 70)

    for query in demo_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)

        # Baseline: rule-based
        baseline_type = tool._classify_query(query)
        print(f"  BASELINE:  route={baseline_type}")

        # RL: bandit routing
        if bandit:
            features = sim.feature_extractor.extract(query)
            arm = bandit.select_arm(features)
            rl_type = ARM_NAMES[arm]
            print(f"  RL BANDIT: route={rl_type}", end="")
            if rl_type != baseline_type:
                print(" [DIFFERENT]", end="")
            print()

            # RL: DQN confidence
            cached = sim.cache.get(query, {})
            results = cached.get(rl_type, [])
            if dqn and results:
                state = ConfidenceState.from_retrieval(results, arm, features)
                action = dqn.select_action(state)
                grade = action_to_grade(action, results[0]["relevance_score"])
                print(f"  RL DQN:    action={ACTION_NAMES[action]}, grade={grade}")

                # Baseline grade
                score = results[0]["relevance_score"] if results else 0
                bl_grade = tool._score_to_grade(score)
                print(f"  BASELINE:  grade={bl_grade}", end="")
                if grade != bl_grade:
                    print(" [DIFFERENT]", end="")
                print()
        else:
            print("  (No trained models found — run 'python -m src.main train' first)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
