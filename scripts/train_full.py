"""Full training pipeline + ablation study."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.training_pipeline import TrainingPipeline
from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS


def main():
    parser = argparse.ArgumentParser(description="PhotoMind RL Training")
    parser.add_argument("--episodes", type=int, default=N_TRAINING_EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()

    pipeline = TrainingPipeline()

    # Full pipeline training
    results = pipeline.train_full(n_episodes=args.episodes, seeds=args.seeds)

    # Optional ablation
    if args.ablation:
        print("\n" + "=" * 60)
        print("ABLATION STUDY")
        print("=" * 60)
        pipeline.run_ablation(n_episodes=args.episodes, seeds=args.seeds)


if __name__ == "__main__":
    main()
