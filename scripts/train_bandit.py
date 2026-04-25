"""Standalone bandit training script."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.training_pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()

    for bandit_type in ["thompson", "ucb", "epsilon_greedy"]:
        print(f"\n{'='*40}")
        print(f"Training {bandit_type} bandit...")
        result = pipeline.train_bandit(bandit_type, n_episodes=2000, seed=42)
        print(f"  Final routing accuracy: {result['final_routing_accuracy']:.1%}")
        print(f"  Cumulative regret: {result['cumulative_regret'][-1]:.1f}")
        print(f"  Arm pulls: {result['arm_pulls']}")


if __name__ == "__main__":
    main()
