"""Plot learning curves for bandit and DQN training."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bandit_regret(training_results: dict, save_dir: str = "./viz/figures"):
    """Plot cumulative regret curves for bandit variants."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # If we have multi-seed data, plot mean + CI
    bandit_data = training_results.get("bandit", {})
    seeds = training_results.get("seeds", [])

    if seeds:
        regrets = []
        for seed in seeds:
            seed_data = bandit_data.get(seed, bandit_data.get(str(seed), {}))
            if "cumulative_regret" in seed_data:
                regrets.append(seed_data["cumulative_regret"])

        if regrets:
            min_len = min(len(r) for r in regrets)
            regrets = np.array([r[:min_len] for r in regrets])
            mean = np.mean(regrets, axis=0)
            std = np.std(regrets, axis=0)
            episodes = np.arange(1, min_len + 1)

            ax.plot(episodes, mean, label="Thompson Sampling", color="#2196F3", linewidth=2)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color="#2196F3")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Bandit Cumulative Regret Over Training")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bandit_regret.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "bandit_regret.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved bandit_regret.png/pdf")


def plot_dqn_rewards(training_results: dict, save_dir: str = "./viz/figures"):
    """Plot DQN episode reward with rolling average."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))

    dqn_data = training_results.get("dqn", {})
    seeds = training_results.get("seeds", [])

    if seeds:
        all_rewards = []
        for seed in seeds:
            seed_data = dqn_data.get(seed, dqn_data.get(str(seed), {}))
            if "rewards" in seed_data:
                all_rewards.append(seed_data["rewards"])

        if all_rewards:
            min_len = min(len(r) for r in all_rewards)
            all_rewards = np.array([r[:min_len] for r in all_rewards])

            # Rolling average
            window = 100
            mean_rewards = np.mean(all_rewards, axis=0)
            rolling = np.convolve(mean_rewards, np.ones(window)/window, mode="valid")
            episodes = np.arange(window, min_len + 1)

            # Raw rewards (faded)
            ax.plot(np.arange(1, min_len + 1), mean_rewards,
                    alpha=0.15, color="#4CAF50", linewidth=0.5)
            # Rolling average
            ax.plot(episodes, rolling, label=f"Rolling Avg (window={window})",
                    color="#4CAF50", linewidth=2)

            # CI band
            std_rewards = np.std(all_rewards, axis=0)
            rolling_std = np.convolve(std_rewards, np.ones(window)/window, mode="valid")
            ax.fill_between(episodes, rolling - rolling_std, rolling + rolling_std,
                            alpha=0.2, color="#4CAF50")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("DQN Confidence Calibrator — Training Reward")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dqn_rewards.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "dqn_rewards.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved dqn_rewards.png/pdf")


def plot_dqn_action_distribution(training_results: dict, save_dir: str = "./viz/figures"):
    """Plot action distribution shift: early vs late training."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    from src.rl.rl_config import ACTION_NAMES

    fig, ax = plt.subplots(figsize=(8, 5))

    dqn_data = training_results.get("dqn", {})
    seeds = training_results.get("seeds", [])

    if seeds:
        # Get rewards from first seed to analyze action distribution
        first_seed = seeds[0]
        seed_data = dqn_data.get(first_seed, dqn_data.get(str(first_seed), {}))
        action_dist = seed_data.get("action_distribution", {})

        if action_dist:
            actions = ACTION_NAMES
            counts = [action_dist.get(i, action_dist.get(str(i), 0)) for i in range(len(actions))]
            total = sum(counts)
            proportions = [c / max(total, 1) for c in counts]

            colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            bars = ax.bar(actions, proportions, color=colors, edgecolor="white", linewidth=1.5)

            for bar, prop in zip(bars, proportions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{prop:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Action")
    ax.set_ylabel("Proportion")
    ax.set_title("DQN Action Distribution (Trained)")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dqn_action_dist.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "dqn_action_dist.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved dqn_action_dist.png/pdf")


def plot_bandit_posteriors(training_results: dict, save_dir: str = "./viz/figures"):
    """Plot Thompson Sampling Beta posterior heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    from src.rl.rl_config import ARM_NAMES

    bandit_data = training_results.get("bandit", {})
    seeds = training_results.get("seeds", [])

    if not seeds:
        return

    first_seed = seeds[0]
    seed_data = bandit_data.get(first_seed, bandit_data.get(str(first_seed), {}))
    posteriors = seed_data.get("posteriors")

    if not posteriors:
        return

    alpha = np.array(posteriors["alpha"])
    beta = np.array(posteriors["beta"])
    means = alpha / (alpha + beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(means, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ARM_NAMES)))
    ax.set_xticklabels(ARM_NAMES)
    ax.set_yticks(range(means.shape[0]))
    ax.set_yticklabels([f"Cluster {i}" for i in range(means.shape[0])])

    # Annotate cells
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means[i,j]:.2f}", ha="center", va="center",
                    color="white" if means[i,j] > 0.5 else "black", fontweight="bold")

    plt.colorbar(im, label="Posterior Mean P(success)")
    ax.set_title("Thompson Sampling: Posterior Mean by Cluster x Strategy")
    ax.set_xlabel("Search Strategy")
    ax.set_ylabel("Query Context Cluster")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bandit_posteriors.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "bandit_posteriors.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved bandit_posteriors.png/pdf")


def plot_epsilon_decay(training_results: dict, save_dir: str = "./viz/figures"):
    """Plot epsilon decay curve for DQN."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(8, 4))

    dqn_data = training_results.get("dqn", {})
    seeds = training_results.get("seeds", [])

    if seeds:
        first_seed = seeds[0]
        seed_data = dqn_data.get(first_seed, dqn_data.get(str(first_seed), {}))
        epsilons = seed_data.get("epsilon_history", [])

        if epsilons:
            ax.plot(range(1, len(epsilons) + 1), epsilons, color="#9C27B0", linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (Exploration Rate)")
    ax.set_title("DQN Epsilon Decay Schedule")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="ε_min = 0.01")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "epsilon_decay.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "epsilon_decay.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved epsilon_decay.png/pdf")


def plot_all(results_path: str = "./eval/results/rl_training_results.json",
             save_dir: str = "./viz/figures"):
    """Generate all learning curve plots from saved training results."""
    with open(results_path) as f:
        results = json.load(f)

    plot_bandit_regret(results, save_dir)
    plot_dqn_rewards(results, save_dir)
    plot_dqn_action_distribution(results, save_dir)
    plot_bandit_posteriors(results, save_dir)
    plot_epsilon_decay(results, save_dir)
    print(f"\nAll learning curve plots saved to {save_dir}/")


if __name__ == "__main__":
    plot_all()
