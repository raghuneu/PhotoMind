"""RL evaluation harness — compares baseline vs RL configurations without API calls."""

import json
import os
import numpy as np

from eval.test_cases import TEST_CASES
from eval.statistical_analysis import confidence_interval, paired_t_test, cohens_d, format_ci
from src.rl.simulation_env import PhotoMindSimulator
from src.rl.contextual_bandit import ThompsonSamplingBandit
from src.rl.dqn_confidence import ConfidenceDQNAgent, ConfidenceState, action_to_grade
from src.rl.rl_config import (
    ARM_NAMES, N_TRAINING_EPISODES, SEEDS, AUGMENTATION_FACTOR,
)
from src.rl.training_pipeline import TrainingPipeline, _set_seed
from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool


def _evaluate_config(sim, test_cases, bandit=None, dqn_agent=None, use_rule_routing=False):
    """Evaluate a single configuration on test cases. Returns per-query results."""
    tool = PhotoKnowledgeBaseTool(knowledge_base_path=sim.kb_path)
    results = []

    for tc in test_cases:
        features = sim.feature_extractor.extract(tc["query"])

        # Routing
        if bandit and not use_rule_routing:
            arm = bandit.select_arm(features)
            strategy = ARM_NAMES[arm]
        else:
            strategy = tool._classify_query(tc["query"])
            arm = ARM_NAMES.index(strategy) if strategy in ARM_NAMES else 1

        # Retrieval from cache
        cached = sim.cache.get(tc["query"], {})
        search_results = cached.get(strategy, [])

        # Check retrieval correctness
        expected_photo = tc.get("expected_photo")
        if expected_photo:
            photo_found = any(
                expected_photo.lower() in r.get("photo_path", "").lower()
                for r in search_results
            )
        else:
            photo_found = True

        # Confidence grading
        if dqn_agent:
            state = ConfidenceState.from_retrieval(search_results, arm, features)
            action = dqn_agent.select_action(state)
            score = search_results[0]["relevance_score"] if search_results else 0.0
            grade = action_to_grade(action, score)
        else:
            score = search_results[0]["relevance_score"] if search_results else 0.0
            if score >= 0.7: grade = "A"
            elif score >= 0.5: grade = "B"
            elif score >= 0.35: grade = "C"
            elif score >= 0.2: grade = "D"
            else: grade = "F"

        # Metrics
        routing_correct = strategy == tc.get("expected_type")
        is_confident = grade in ("A", "B", "C")
        silent_failure = not photo_found and is_confident and expected_photo is not None
        declined_correctly = None
        if tc.get("should_decline"):
            declined_correctly = grade in ("D", "F")

        results.append({
            "query": tc["query"],
            "category": tc.get("category"),
            "retrieval_correct": photo_found,
            "routing_correct": routing_correct,
            "silent_failure": silent_failure,
            "declined_correctly": declined_correctly,
            "confidence_grade": grade,
            "strategy_used": strategy,
        })

    return results


def _aggregate_results(results):
    """Compute aggregate metrics from per-query results."""
    total = len(results)
    retrieval_acc = sum(r["retrieval_correct"] for r in results) / total
    routing_acc = sum(r["routing_correct"] for r in results) / total
    silent_failures = sum(1 for r in results if r["silent_failure"])
    silent_failure_rate = silent_failures / total

    decline_cases = [r for r in results if r["declined_correctly"] is not None]
    decline_acc = (
        sum(r["declined_correctly"] for r in decline_cases) / len(decline_cases)
        if decline_cases else 0.0
    )

    return {
        "retrieval_accuracy": retrieval_acc,
        "routing_accuracy": routing_acc,
        "silent_failure_rate": silent_failure_rate,
        "decline_accuracy": decline_acc,
    }


def run_rl_eval(n_episodes: int = N_TRAINING_EPISODES, seeds: list | None = None):
    """Run 4-config RL evaluation with statistical analysis."""
    if seeds is None:
        seeds = SEEDS

    # Try to load expanded test cases
    try:
        from eval.expanded_test_cases import ALL_TEST_CASES
        test_cases = ALL_TEST_CASES
        print(f"Using expanded test suite: {len(test_cases)} queries")
    except ImportError:
        test_cases = TEST_CASES
        print(f"Using original test suite: {len(test_cases)} queries")

    configs = [
        {"name": "Baseline (Rule-Based)", "bandit": False, "dqn": False, "rule_routing": True},
        {"name": "Bandit Only (Thompson)", "bandit": True, "dqn": False, "rule_routing": False},
        {"name": "DQN Only", "bandit": False, "dqn": True, "rule_routing": True},
        {"name": "Full RL (Thompson+DQN)", "bandit": True, "dqn": True, "rule_routing": False},
    ]

    print(f"\nRL Evaluation: {len(configs)} configs x {len(seeds)} seeds")
    print("=" * 70)

    all_config_results = {}
    pipeline = TrainingPipeline(test_cases=test_cases)

    for config in configs:
        config_name = config["name"]
        per_seed_metrics = []

        for seed in seeds:
            _set_seed(seed)
            sim = PhotoMindSimulator(pipeline.kb_path, test_cases, augmentation_factor=1)

            trained_bandit = None
            if config["bandit"]:
                bandit_result = pipeline.train_bandit("thompson", n_episodes, seed)
                trained_bandit = bandit_result["_bandit"]

            trained_dqn = None
            if config["dqn"]:
                dqn_result = pipeline.train_dqn(n_episodes, seed, trained_bandit)
                trained_dqn = dqn_result["_agent"]
                trained_dqn.epsilon = 0.0

            results = _evaluate_config(
                sim, test_cases,
                bandit=trained_bandit,
                dqn_agent=trained_dqn,
                use_rule_routing=config["rule_routing"],
            )
            metrics = _aggregate_results(results)
            per_seed_metrics.append(metrics)

        # Aggregate across seeds
        aggregated = {}
        for metric in ["retrieval_accuracy", "routing_accuracy",
                       "silent_failure_rate", "decline_accuracy"]:
            values = [m[metric] for m in per_seed_metrics]
            mean, lower, upper, margin = confidence_interval(values)
            aggregated[metric] = {
                "mean": mean, "lower": lower, "upper": upper,
                "values": values,
            }

        all_config_results[config_name] = {
            "per_seed": per_seed_metrics,
            "aggregated": aggregated,
        }

    # Print comparison table — column width 24 accommodates "87.5% [85.3%, 89.7%]"
    print(f"\n{'Config':<30} {'Retrieval':>24} {'Routing':>24} {'SilentFail':>24} {'Decline':>24}")
    print("-" * 106)
    for name, data in all_config_results.items():
        agg = data["aggregated"]
        print(f"{name:<30} "
              f"{format_ci(agg['retrieval_accuracy']['mean'], agg['retrieval_accuracy']['lower'], agg['retrieval_accuracy']['upper']):>24} "
              f"{format_ci(agg['routing_accuracy']['mean'], agg['routing_accuracy']['lower'], agg['routing_accuracy']['upper']):>24} "
              f"{format_ci(agg['silent_failure_rate']['mean'], agg['silent_failure_rate']['lower'], agg['silent_failure_rate']['upper']):>24} "
              f"{format_ci(agg['decline_accuracy']['mean'], agg['decline_accuracy']['lower'], agg['decline_accuracy']['upper']):>24}")

    # Statistical tests: Full RL vs Baseline
    baseline = all_config_results["Baseline (Rule-Based)"]
    full_rl = all_config_results["Full RL (Thompson+DQN)"]

    print("\n--- Statistical Tests: Full RL vs Baseline ---")
    for metric in ["retrieval_accuracy", "routing_accuracy",
                   "silent_failure_rate", "decline_accuracy"]:
        bl_vals = baseline["aggregated"][metric]["values"]
        rl_vals = full_rl["aggregated"][metric]["values"]
        t_stat, p_val = paired_t_test(bl_vals, rl_vals)
        d = cohens_d(bl_vals, rl_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        # Cohen's d is undefined (inf) when variance collapses to zero but mean differs —
        # this represents a deterministic, consistent effect across all seeds.
        d_str = "inf*" if d == float('inf') else f"{d:>6.3f}"
        print(f"  {metric:<25} t={t_stat:>7.3f}, p={p_val:.4f} {sig}, d={d_str}")

    # Save results
    results_path = "./eval/results/rl_eval_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_config_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_config_results


if __name__ == "__main__":
    run_rl_eval()
