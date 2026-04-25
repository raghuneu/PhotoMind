"""Plot ablation study results as grouped bar chart."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eval.statistical_analysis import confidence_interval


def plot_ablation(results_path: str = "./eval/results/ablation_results.json",
                  save_dir: str = "./viz/figures"):
    """Generate grouped bar chart for ablation results."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    with open(results_path) as f:
        results = json.load(f)

    metrics = ["retrieval_accuracy", "routing_accuracy", "silent_failure_rate", "decline_accuracy"]
    metric_labels = ["Retrieval\nAccuracy", "Routing\nAccuracy", "Silent Failure\nRate", "Decline\nAccuracy"]

    config_names = list(results.keys())
    n_configs = len(config_names)
    n_metrics = len(metrics)

    # Short names for x-axis
    short_names = []
    for name in config_names:
        if "Full" in name:
            short_names.append("Full RL")
        elif "Bandit Only" in name:
            short_names.append("Bandit\nOnly")
        elif "DQN Only" in name:
            short_names.append("DQN\nOnly")
        elif "Thompson" in name and "Full" not in name:
            short_names.append("Thompson")
        elif "UCB" in name:
            short_names.append("UCB")
        elif "Epsilon" in name:
            short_names.append("ε-Greedy")
        elif "Baseline" in name:
            short_names.append("Baseline")
        else:
            short_names.append(name[:12])

    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5), sharey=False)

    colors = sns.color_palette("Set2", n_configs)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        means = []
        errors = []

        for config_name in config_names:
            agg = results[config_name]["aggregated"][metric]
            values = agg.get("values", [agg["mean"]])
            mean, lower, upper, margin = confidence_interval(values)
            means.append(mean)
            errors.append(margin)

        x = np.arange(n_configs)
        bars = ax.bar(x, means, yerr=errors, capsize=3, color=colors,
                       edgecolor="white", linewidth=1.5, error_kw={"linewidth": 1.5})

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_ylim(0, 1.05)

        # Annotate bars
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{mean_val:.0%}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.suptitle("Ablation Study: Component Contribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "ablation_comparison.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved ablation_comparison.png/pdf")


if __name__ == "__main__":
    plot_ablation()
