"""Plot cumulative regret curves comparing bandit algorithms."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eval.test_cases import TEST_CASES
from src.rl.training_pipeline import TrainingPipeline, _set_seed
from src.rl.rl_config import SEEDS


def plot_regret_comparison(n_episodes: int = 2000, seeds: list | None = None,
                           save_dir: str = "./viz/figures"):
    """Train all 3 bandit types and compare cumulative regret."""
    if seeds is None:
        seeds = SEEDS
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    pipeline = TrainingPipeline()
    bandit_types = ["thompson", "ucb", "epsilon_greedy"]
    labels = ["Thompson Sampling", "UCB1", "ε-Greedy"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for bt, label, color in zip(bandit_types, labels, colors):
        all_regrets = []
        for seed in seeds:
            result = pipeline.train_bandit(bt, n_episodes, seed)
            all_regrets.append(result["cumulative_regret"])

        min_len = min(len(r) for r in all_regrets)
        all_regrets = np.array([r[:min_len] for r in all_regrets])
        mean = np.mean(all_regrets, axis=0)
        std = np.std(all_regrets, axis=0)
        episodes = np.arange(1, min_len + 1)

        ax.plot(episodes, mean, label=label, color=color, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret: Thompson Sampling vs UCB vs ε-Greedy")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regret_comparison.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "regret_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved regret_comparison.png/pdf")


def plot_routing_accuracy_over_time(n_episodes: int = 2000, seeds: list | None = None,
                                     save_dir: str = "./viz/figures"):
    """Plot routing accuracy convergence over training."""
    if seeds is None:
        seeds = SEEDS
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    pipeline = TrainingPipeline()
    bandit_types = ["thompson", "ucb", "epsilon_greedy"]
    labels = ["Thompson Sampling", "UCB1", "ε-Greedy"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for bt, label, color in zip(bandit_types, labels, colors):
        all_accs = []
        for seed in seeds:
            result = pipeline.train_bandit(bt, n_episodes, seed)
            all_accs.append(result["routing_accuracy_history"])

        min_len = min(len(a) for a in all_accs)
        all_accs = np.array([a[:min_len] for a in all_accs])
        mean = np.mean(all_accs, axis=0)
        std = np.std(all_accs, axis=0)
        checkpoints = np.arange(1, min_len + 1) * 100

        ax.plot(checkpoints, mean, label=label, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(checkpoints, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Routing Accuracy")
    ax.set_title("Routing Accuracy Convergence Over Training")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "routing_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "routing_accuracy.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved routing_accuracy.png/pdf")


if __name__ == "__main__":
    plot_regret_comparison()
    plot_routing_accuracy_over_time()
