"""RL evaluation harness — compares baseline vs RL configurations without API calls."""

import json
import os
import numpy as np

from eval.test_cases import TEST_CASES
from eval.statistical_analysis import confidence_interval, paired_t_test, cohens_d, format_ci
from src.rl.simulation_env import PhotoMindSimulator
from src.rl.contextual_bandit import ThompsonSamplingBandit
from src.rl.dqn_confidence import ConfidenceDQNAgent, ConfidenceState, action_to_grade, resolve_confidence_grade
from src.rl.rl_config import (
    ARM_NAMES, N_TRAINING_EPISODES, SEEDS, AUGMENTATION_FACTOR,
    REQUERY_ACTION, MAX_REQUERY_STEPS,
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
            grade, _ = resolve_confidence_grade(dqn_agent, search_results, arm, features, cached)
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

    # Tool usage distribution
    strategy_counts = {}
    grade_counts = {}
    for r in results:
        s = r.get("strategy_used", "unknown")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        g = r.get("confidence_grade", "?")
        grade_counts[g] = grade_counts.get(g, 0) + 1

    return {
        "retrieval_accuracy": retrieval_acc,
        "routing_accuracy": routing_acc,
        "silent_failure_rate": silent_failure_rate,
        "decline_accuracy": decline_acc,
        "strategy_distribution": strategy_counts,
        "grade_distribution": grade_counts,
    }


def run_rl_eval(n_episodes: int = N_TRAINING_EPISODES, seeds: list | None = None):
    """Run 5-config RL evaluation with statistical analysis and held-out generalization."""
    if seeds is None:
        seeds = SEEDS

    # Load test suites
    try:
        from eval.expanded_test_cases import ALL_TEST_CASES, TRAIN_TEST_CASES, HELD_OUT_TEST_CASES
        full_test_cases = ALL_TEST_CASES
        train_test_cases = TRAIN_TEST_CASES
        held_out_test_cases = HELD_OUT_TEST_CASES
        print(f"Test suites: {len(full_test_cases)} total, "
              f"{len(train_test_cases)} train, {len(held_out_test_cases)} held-out")
    except ImportError:
        full_test_cases = TEST_CASES
        train_test_cases = TEST_CASES
        held_out_test_cases = []
        print(f"Using original test suite: {len(full_test_cases)} queries (no held-out split)")

    configs = [
        {"name": "Baseline (Rule-Based)", "bandit": False, "dqn": False, "rule_routing": True, "deterministic": True},
        {"name": "Bandit Only (Thompson)", "bandit": True, "dqn": False, "rule_routing": False},
        {"name": "DQN Only", "bandit": False, "dqn": True, "rule_routing": True},
        {"name": "Full RL (Thompson+DQN)", "bandit": True, "dqn": True, "rule_routing": False},
        {"name": "Recommended (Rule+DQN)", "bandit": False, "dqn": True, "rule_routing": True},
    ]

    print(f"\nRL Evaluation: {len(configs)} configs x {len(seeds)} seeds")
    print("=" * 70)

    all_config_results = {}
    # Train on train split only to prevent data leakage
    pipeline = TrainingPipeline(test_cases=train_test_cases)

    for config in configs:
        config_name = config["name"]
        is_deterministic = config.get("deterministic", False)
        # Deterministic configs (rule-based routing + no RL) produce identical
        # metrics across seeds by construction — run once and label as such
        # instead of reporting 5 duplicate rows as if they demonstrated variance.
        effective_seeds = [seeds[0]] if is_deterministic else seeds
        per_seed_metrics = {"full": [], "held_out": []}

        for seed in effective_seeds:
            _set_seed(seed)

            trained_bandit = None
            if config["bandit"]:
                bandit_result = pipeline.train_bandit("thompson", n_episodes, seed)
                trained_bandit = bandit_result["_bandit"]

            trained_dqn = None
            if config["dqn"]:
                dqn_result = pipeline.train_dqn(n_episodes, seed, trained_bandit)
                trained_dqn = dqn_result["_agent"]
                trained_dqn.epsilon = 0.0

            # Evaluate on full set
            sim_full = PhotoMindSimulator(pipeline.kb_path, full_test_cases, augmentation_factor=1)
            results_full = _evaluate_config(
                sim_full, full_test_cases,
                bandit=trained_bandit,
                dqn_agent=trained_dqn,
                use_rule_routing=config["rule_routing"],
            )
            per_seed_metrics["full"].append(_aggregate_results(results_full))

            # Evaluate on held-out set (generalization)
            if held_out_test_cases:
                sim_ho = PhotoMindSimulator(pipeline.kb_path, held_out_test_cases, augmentation_factor=1)
                results_ho = _evaluate_config(
                    sim_ho, held_out_test_cases,
                    bandit=trained_bandit,
                    dqn_agent=trained_dqn,
                    use_rule_routing=config["rule_routing"],
                )
                per_seed_metrics["held_out"].append(_aggregate_results(results_ho))

        # Aggregate across seeds for each split
        aggregated = {}
        for split_name, seed_metrics in per_seed_metrics.items():
            if not seed_metrics:
                continue
            split_agg = {}
            for metric in ["retrieval_accuracy", "routing_accuracy",
                           "silent_failure_rate", "decline_accuracy"]:
                values = [m[metric] for m in seed_metrics]
                mean, lower, upper, margin = confidence_interval(values)
                split_agg[metric] = {
                    "mean": mean, "lower": lower, "upper": upper,
                    "values": values,
                }
            aggregated[split_name] = split_agg

        all_config_results[config_name] = {
            "per_seed": per_seed_metrics,
            "aggregated": aggregated,
            "deterministic": is_deterministic,
            "n_seeds": len(effective_seeds),
        }

    # Print comparison table — Full test set
    print(f"\n=== Full Test Set ({len(full_test_cases)} queries) ===")
    print(f"{'Config':<30} {'Retrieval':>24} {'Routing':>24} {'SilentFail':>24} {'Decline':>24}")
    print("-" * 130)
    print("[det.] = deterministic (n=1); non-det. configs report mean over "
          f"{len(seeds)} seeds with 95% CI.")
    print("-" * 130)
    for name, data in all_config_results.items():
        agg = data["aggregated"]["full"]
        display_name = f"{name} [det.]" if data.get("deterministic") else name
        print(f"{display_name:<30} "
              f"{format_ci(agg['retrieval_accuracy']['mean'], agg['retrieval_accuracy']['lower'], agg['retrieval_accuracy']['upper']):>24} "
              f"{format_ci(agg['routing_accuracy']['mean'], agg['routing_accuracy']['lower'], agg['routing_accuracy']['upper']):>24} "
              f"{format_ci(agg['silent_failure_rate']['mean'], agg['silent_failure_rate']['lower'], agg['silent_failure_rate']['upper']):>24} "
              f"{format_ci(agg['decline_accuracy']['mean'], agg['decline_accuracy']['lower'], agg['decline_accuracy']['upper']):>24}")

    # Print comparison table — Held-out set
    if held_out_test_cases:
        print(f"\n=== Held-Out Generalization ({len(held_out_test_cases)} queries) ===")
        print(f"{'Config':<30} {'Retrieval':>24} {'Routing':>24} {'SilentFail':>24} {'Decline':>24}")
        print("-" * 130)
        for name, data in all_config_results.items():
            agg = data["aggregated"].get("held_out", {})
            if not agg:
                continue
            print(f"{name:<30} "
                  f"{format_ci(agg['retrieval_accuracy']['mean'], agg['retrieval_accuracy']['lower'], agg['retrieval_accuracy']['upper']):>24} "
                  f"{format_ci(agg['routing_accuracy']['mean'], agg['routing_accuracy']['lower'], agg['routing_accuracy']['upper']):>24} "
                  f"{format_ci(agg['silent_failure_rate']['mean'], agg['silent_failure_rate']['lower'], agg['silent_failure_rate']['upper']):>24} "
                  f"{format_ci(agg['decline_accuracy']['mean'], agg['decline_accuracy']['lower'], agg['decline_accuracy']['upper']):>24}")

    # Statistical tests: Full RL vs Baseline (on full set)
    baseline = all_config_results["Baseline (Rule-Based)"]
    full_rl = all_config_results["Full RL (Thompson+DQN)"]

    # Tool usage summary (from first seed of each config on full set)
    print("\n=== Tool Usage Summary (Strategy & Grade Distribution) ===")
    for name in all_config_results:
        data = all_config_results.get(name)
        if not data:
            continue
        full_seeds = data["per_seed"].get("full", data["per_seed"])
        if isinstance(full_seeds, list) and full_seeds:
            first = full_seeds[0]
            strat = first.get("strategy_distribution", {})
            grade = first.get("grade_distribution", {})
            print(f"  {name:<30} strategies={strat}  grades={grade}")

    print("\n--- Statistical Tests: Full RL vs Baseline (Full Set) ---")
    for metric in ["retrieval_accuracy", "routing_accuracy",
                   "silent_failure_rate", "decline_accuracy"]:
        bl_vals = baseline["aggregated"]["full"][metric]["values"]
        rl_vals = full_rl["aggregated"]["full"][metric]["values"]
        t_stat, p_val = paired_t_test(bl_vals, rl_vals)
        d = cohens_d(bl_vals, rl_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        d_str = "inf*" if d == float('inf') else f"{d:>6.3f}"
        print(f"  {metric:<25} t={t_stat:>7.3f}, p={p_val:.4f} {sig}, d={d_str}")

    # Statistical tests: Recommended vs Baseline (on held-out set)
    if held_out_test_cases:
        recommended = all_config_results["Recommended (Rule+DQN)"]
        print("\n--- Statistical Tests: Recommended (Rule+DQN) vs Baseline (Held-Out) ---")
        for metric in ["retrieval_accuracy", "routing_accuracy",
                       "silent_failure_rate", "decline_accuracy"]:
            bl_vals = baseline["aggregated"]["held_out"][metric]["values"]
            rec_vals = recommended["aggregated"]["held_out"][metric]["values"]
            t_stat, p_val = paired_t_test(bl_vals, rec_vals)
            d = cohens_d(bl_vals, rec_vals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
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
