"""Plot before/after learning progress — shows RL improvement over baseline.

Generates a side-by-side bar chart comparing Baseline vs Full RL vs Recommended
configurations across all metrics, with held-out generalization results.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eval.statistical_analysis import confidence_interval


def plot_before_after(results_path: str = "./eval/results/rl_eval_results.json",
                      save_dir: str = "./viz/figures"):
    """Generate before/after learning progress figure."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    with open(results_path) as f:
        results = json.load(f)

    # Configs to compare
    config_keys = ["Baseline (Rule-Based)", "Full RL (Thompson+DQN)", "Recommended (Rule+DQN)"]
    short_labels = ["Baseline", "Full RL", "Recommended\n(Rule+DQN)"]
    colors = ["#95a5a6", "#e74c3c", "#2ecc71"]

    metrics = ["retrieval_accuracy", "routing_accuracy", "silent_failure_rate", "decline_accuracy"]
    metric_labels = ["Retrieval\nAccuracy", "Routing\nAccuracy", "Silent Failure\nRate", "Decline\nAccuracy"]

    # Determine which splits are available
    splits = []
    sample_config = results.get(config_keys[0], {}).get("aggregated", {})
    if "full" in sample_config:
        splits.append("full")
    if "held_out" in sample_config:
        splits.append("held_out")

    # Fall back: if no "full"/"held_out" keys, data uses old flat format
    if not splits:
        splits.append("flat")

    n_splits = len(splits)
    fig, axes = plt.subplots(n_splits, len(metrics), figsize=(16, 5 * n_splits),
                             squeeze=False)

    for row, split in enumerate(splits):
        split_title = "Full Test Set" if split == "full" else (
            "Held-Out Generalization" if split == "held_out" else "All Queries"
        )

        for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row][col]
            means = []
            errors = []

            for config_key in config_keys:
                if config_key not in results:
                    means.append(0.0)
                    errors.append(0.0)
                    continue

                agg = results[config_key].get("aggregated", {})
                if split == "flat":
                    metric_data = agg.get(metric, {})
                else:
                    metric_data = agg.get(split, {}).get(metric, {})

                values = metric_data.get("values", [metric_data.get("mean", 0.0)])
                if isinstance(values, (int, float)):
                    values = [values]
                mean, lower, upper, margin = confidence_interval(values)
                means.append(mean)
                errors.append(margin)

            x = np.arange(len(config_keys))
            bars = ax.bar(x, means, yerr=errors, capsize=4,
                         color=colors, edgecolor="white", linewidth=1.5,
                         error_kw={"linewidth": 1.5})

            ax.set_title(label, fontweight="bold", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, fontsize=8)
            ax.set_ylim(0, 1.15)

            # Annotate bars
            for bar, mean_val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.03,
                       f"{mean_val:.1%}", ha="center", va="bottom",
                       fontsize=9, fontweight="bold")

            # Highlight silent failure improvement with arrow
            if metric == "silent_failure_rate" and len(means) >= 2:
                baseline_val = means[0]
                rl_val = means[1]
                if baseline_val > rl_val:
                    ax.annotate("",
                               xy=(1, rl_val + 0.01),
                               xytext=(0, baseline_val + 0.01),
                               arrowprops=dict(arrowstyle="->", color="#27ae60",
                                             lw=2, connectionstyle="arc3,rad=-0.3"))

            if col == 0:
                ax.set_ylabel(split_title, fontsize=11, fontweight="bold")

    fig.suptitle("Learning Progress: Before vs After RL Training",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "before_after_learning.png"),
               dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "before_after_learning.pdf"),
               bbox_inches="tight")
    plt.close()
    print("Saved before_after_learning.png/pdf")


if __name__ == "__main__":
    plot_before_after()
