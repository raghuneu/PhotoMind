"""Ablation study runner — 7 configs x 5 seeds with statistical validation."""

import json
import os
import numpy as np

from eval.test_cases import TEST_CASES
from eval.statistical_analysis import confidence_interval, paired_t_test, cohens_d, format_ci
from src.rl.training_pipeline import TrainingPipeline
from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS


def run_ablation(n_episodes: int = N_TRAINING_EPISODES, seeds: list | None = None):
    """Run full ablation study and produce comparison table."""
    if seeds is None:
        seeds = SEEDS

    # Try expanded test cases
    try:
        from eval.expanded_test_cases import ALL_TEST_CASES
        test_cases = ALL_TEST_CASES
    except ImportError:
        test_cases = TEST_CASES

    print(f"Ablation Study: {len(test_cases)} test cases, {len(seeds)} seeds")
    print("=" * 70)

    pipeline = TrainingPipeline(test_cases=test_cases)
    results = pipeline.run_ablation(n_episodes=n_episodes, seeds=seeds)

    # Print formatted table with CIs
    print(f"\n{'='*90}")
    print(f"ABLATION RESULTS (mean [95% CI])")
    print(f"{'='*90}")
    print(f"{'Config':<30} {'Retrieval':>14} {'Routing':>14} {'SilentFail':>14} {'Decline':>14}")
    print("-" * 86)

    baseline_metrics = None
    for config_name, data in results.items():
        agg = data["aggregated"]
        row = f"{config_name:<30}"
        for metric in ["retrieval_accuracy", "routing_accuracy",
                       "silent_failure_rate", "decline_accuracy"]:
            vals = agg[metric]["values"]
            mean, lower, upper, _ = confidence_interval(vals)
            row += f" {mean:>5.1%}[{lower:.1%},{upper:.1%}]"
        print(row)

        if "Baseline" in config_name:
            baseline_metrics = agg

    # Statistical comparisons vs baseline
    if baseline_metrics:
        print(f"\n--- Paired t-tests vs Baseline ---")
        print(f"{'Config':<30} {'Metric':<25} {'t':>8} {'p':>8} {'sig':>4} {'d':>8}")
        print("-" * 83)
        for config_name, data in results.items():
            if "Baseline" in config_name:
                continue
            for metric in ["retrieval_accuracy", "routing_accuracy",
                           "silent_failure_rate", "decline_accuracy"]:
                bl_vals = baseline_metrics[metric]["values"]
                rl_vals = data["aggregated"][metric]["values"]
                t_stat, p_val = paired_t_test(bl_vals, rl_vals)
                d = cohens_d(bl_vals, rl_vals)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                # Cohen's d is inf when variance collapses to zero — represents a deterministic effect.
                d_str = "     inf*" if d == float('inf') else f"{d:>8.3f}"
                print(f"{config_name:<30} {metric:<25} {t_stat:>8.3f} {p_val:>8.4f} {sig:>4} {d_str}")

    # Save
    output_path = "./eval/results/ablation_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_ablation()
