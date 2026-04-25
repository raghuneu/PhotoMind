"""Standalone DQN training script."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.training_pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()

    print("Training DQN confidence calibrator...")
    result = pipeline.train_dqn(n_episodes=2000, seed=42)
    print(f"  Avg reward (last 100): {result['avg_reward_last_100']:.3f}")
    print(f"  Action distribution: {result['action_distribution']}")
    print(f"  Final epsilon: {result['epsilon_history'][-1]:.4f}")


if __name__ == "__main__":
    main()
