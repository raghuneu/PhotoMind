"""Training pipeline — orchestrates bandit and DQN training across seeds."""

import json
import os
import time
import numpy as np
from collections import deque

from src.rl.rl_config import (
    ARM_NAMES, N_TRAINING_EPISODES, SEEDS, AUGMENTATION_FACTOR,
    BANDIT_MODEL_PATH, DQN_MODEL_PATH, ACTION_NAMES,
)
from src.rl.simulation_env import PhotoMindSimulator
from src.rl.contextual_bandit import (
    ThompsonSamplingBandit, UCBBandit, EpsilonGreedyBandit,
)
from src.rl.dqn_confidence import (
    ConfidenceDQNAgent, ConfidenceState, action_to_grade,
)


def _set_seed(seed: int):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainingPipeline:
    """Orchestrates RL training for PhotoMind."""

    def __init__(self, kb_path: str = "./knowledge_base/photo_index.json",
                 test_cases: list | None = None):
        if test_cases is None:
            from eval.test_cases import TEST_CASES
            test_cases = TEST_CASES
        self.kb_path = kb_path
        self.test_cases = test_cases

    def train_bandit(self, bandit_type: str = "thompson",
                     n_episodes: int = N_TRAINING_EPISODES,
                     seed: int = 42) -> dict:
        """Train a contextual bandit and return metrics."""
        _set_seed(seed)

        sim = PhotoMindSimulator(self.kb_path, self.test_cases,
                                 augmentation_factor=AUGMENTATION_FACTOR)
        all_features = sim.get_all_features()

        # Create bandit
        if bandit_type == "thompson":
            bandit = ThompsonSamplingBandit()
        elif bandit_type == "ucb":
            bandit = UCBBandit()
        elif bandit_type == "epsilon_greedy":
            bandit = EpsilonGreedyBandit()
        else:
            raise ValueError(f"Unknown bandit type: {bandit_type}")

        bandit.fit_clusters(all_features)

        # Training loop
        cumulative_regret = 0.0
        regret_history = []
        arm_pulls = {a: 0 for a in range(len(ARM_NAMES))}
        rewards_history = []
        routing_accuracy_history = []

        for ep in range(n_episodes):
            features, info = sim.reset()
            arm = bandit.select_arm(features)
            results, reward, binfo = sim.step_bandit(arm)
            bandit.update(features, arm, reward)

            # Track metrics
            arm_pulls[arm] += 1
            rewards_history.append(reward)
            optimal_reward = 1.0
            cumulative_regret += (optimal_reward - reward)
            regret_history.append(cumulative_regret)

            # Periodic routing accuracy check
            if (ep + 1) % 100 == 0:
                correct = 0
                total = 0
                for tc in self.test_cases:
                    if tc.get("should_decline"):
                        continue
                    f = sim.feature_extractor.extract(tc["query"])
                    a = bandit.select_arm(f)
                    if ARM_NAMES[a] == tc.get("expected_type"):
                        correct += 1
                    total += 1
                routing_accuracy_history.append(correct / max(total, 1))

        # Posteriors for Thompson Sampling visualization
        posteriors = None
        if bandit_type == "thompson":
            posteriors = {
                "alpha": bandit.alpha.tolist(),
                "beta": bandit.beta_param.tolist(),
            }

        return {
            "bandit_type": bandit_type,
            "seed": seed,
            "n_episodes": n_episodes,
            "cumulative_regret": regret_history,
            "arm_pulls": arm_pulls,
            "rewards": rewards_history,
            "routing_accuracy_history": routing_accuracy_history,
            "final_routing_accuracy": routing_accuracy_history[-1] if routing_accuracy_history else 0.0,
            "posteriors": posteriors,
            "_bandit": bandit,
        }

    def train_dqn(self, n_episodes: int = N_TRAINING_EPISODES,
                  seed: int = 42, trained_bandit=None) -> dict:
        """Train the DQN confidence calibrator and return metrics."""
        _set_seed(seed)

        sim = PhotoMindSimulator(self.kb_path, self.test_cases,
                                 augmentation_factor=AUGMENTATION_FACTOR)
        agent = ConfidenceDQNAgent()

        rewards_history = []
        loss_history = []
        epsilon_history = []
        action_counts = {a: 0 for a in range(len(ACTION_NAMES))}
        window = deque(maxlen=100)

        # Rule-based router for DQN-only training (avoids oracle distribution mismatch)
        from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool as _PKBToolDQN
        _dqn_rule_tool = _PKBToolDQN(knowledge_base_path=self.kb_path)

        for ep in range(n_episodes):
            features, info = sim.reset()

            # Select strategy: use trained bandit, rule-based router, or oracle fallback
            if trained_bandit is not None:
                arm = trained_bandit.select_arm(features)
            else:
                # Use rule-based routing to avoid training/evaluation distribution mismatch
                rule_strategy = _dqn_rule_tool._classify_query(info["query"])
                arm = ARM_NAMES.index(rule_strategy) if rule_strategy in ARM_NAMES else 1

            results, _, _ = sim.step_bandit(arm)
            state = ConfidenceState.from_retrieval(results, arm, features)
            action = agent.select_action(state)
            reward, done, dinfo = sim.step_confidence(action, results, info)

            # Single-step episode: next_state = terminal state (zeros)
            next_state = np.zeros_like(state)
            loss = agent.step(state, action, reward, next_state, done)

            agent.decay_epsilon()

            # Track metrics
            rewards_history.append(reward)
            window.append(reward)
            action_counts[action] += 1
            epsilon_history.append(agent.epsilon)
            if loss is not None:
                loss_history.append(loss)

        return {
            "seed": seed,
            "n_episodes": n_episodes,
            "rewards": rewards_history,
            "avg_reward_last_100": float(np.mean(list(window))),
            "loss_history": loss_history,
            "epsilon_history": epsilon_history,
            "action_distribution": action_counts,
            "_agent": agent,
        }

    def train_full(self, n_episodes: int = N_TRAINING_EPISODES,
                   seeds: list | None = None) -> dict:
        """Train both components across multiple seeds."""
        if seeds is None:
            seeds = SEEDS

        print(f"Training full pipeline: {n_episodes} episodes x {len(seeds)} seeds")
        print("=" * 60)

        all_results = {"bandit": {}, "dqn": {}, "seeds": seeds}
        best_bandit = None
        best_bandit_acc = 0.0

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")

            # Train bandit
            start = time.time()
            bandit_result = self.train_bandit("thompson", n_episodes, seed)
            elapsed = time.time() - start
            acc = bandit_result["final_routing_accuracy"]
            print(f"  Bandit: routing_acc={acc:.1%}, "
                  f"regret={bandit_result['cumulative_regret'][-1]:.1f}, "
                  f"{elapsed:.1f}s")
            all_results["bandit"][seed] = {
                k: v for k, v in bandit_result.items() if k != "_bandit"
            }

            if acc > best_bandit_acc:
                best_bandit_acc = acc
                best_bandit = bandit_result["_bandit"]

            # Train DQN using trained bandit
            start = time.time()
            dqn_result = self.train_dqn(n_episodes, seed, bandit_result["_bandit"])
            elapsed = time.time() - start
            print(f"  DQN: avg_reward={dqn_result['avg_reward_last_100']:.3f}, "
                  f"actions={dqn_result['action_distribution']}, "
                  f"{elapsed:.1f}s")
            all_results["dqn"][seed] = {
                k: v for k, v in dqn_result.items() if k != "_agent"
            }

            # Save best DQN from last seed
            best_dqn_agent = dqn_result["_agent"]

        # Save best models
        if best_bandit:
            best_bandit.save(BANDIT_MODEL_PATH)
            print(f"\nBandit model saved: {BANDIT_MODEL_PATH}")
        if best_dqn_agent:
            best_dqn_agent.save(DQN_MODEL_PATH)
            print(f"DQN model saved: {DQN_MODEL_PATH}")

        # Aggregate statistics
        bandit_accs = [all_results["bandit"][s]["final_routing_accuracy"] for s in seeds]
        dqn_rewards = [all_results["dqn"][s]["avg_reward_last_100"] for s in seeds]

        all_results["summary"] = {
            "bandit_routing_accuracy": {
                "mean": float(np.mean(bandit_accs)),
                "std": float(np.std(bandit_accs)),
                "per_seed": {s: a for s, a in zip(seeds, bandit_accs)},
            },
            "dqn_avg_reward": {
                "mean": float(np.mean(dqn_rewards)),
                "std": float(np.std(dqn_rewards)),
                "per_seed": {s: r for s, r in zip(seeds, dqn_rewards)},
            },
        }

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"  Bandit routing accuracy: {np.mean(bandit_accs):.1%} +/- {np.std(bandit_accs):.1%}")
        print(f"  DQN avg reward (last 100): {np.mean(dqn_rewards):.3f} +/- {np.std(dqn_rewards):.3f}")

        # Save results
        results_path = "./eval/results/rl_training_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {results_path}")

        return all_results

    def run_ablation(self, n_episodes: int = N_TRAINING_EPISODES,
                     seeds: list | None = None) -> dict:
        """Run 7-config ablation study across all seeds."""
        from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool as _PKBTool
        if seeds is None:
            seeds = SEEDS

        configs = [
            {"name": "Full (Thompson+DQN)", "bandit": "thompson", "dqn": True},
            {"name": "Bandit Only (Thompson)", "bandit": "thompson", "dqn": False},
            {"name": "DQN Only", "bandit": None, "dqn": True},
            {"name": "UCB + DQN", "bandit": "ucb", "dqn": True},
            {"name": "UCB", "bandit": "ucb", "dqn": False},
            {"name": "Epsilon-Greedy", "bandit": "epsilon_greedy", "dqn": False},
            {"name": "Baseline (Rule-Based)", "bandit": None, "dqn": False},
        ]

        print(f"Ablation Study: {len(configs)} configs x {len(seeds)} seeds")
        print("=" * 60)

        results = {}
        for config in configs:
            config_name = config["name"]
            print(f"\n--- {config_name} ---")
            config_results = []

            for seed in seeds:
                _set_seed(seed)
                sim = PhotoMindSimulator(self.kb_path, self.test_cases, augmentation_factor=1)

                # Train bandit if specified
                trained_bandit = None
                if config["bandit"]:
                    bandit_result = self.train_bandit(config["bandit"], n_episodes, seed)
                    trained_bandit = bandit_result["_bandit"]

                # Train DQN if specified
                trained_dqn = None
                if config["dqn"]:
                    dqn_result = self.train_dqn(n_episodes, seed, trained_bandit)
                    trained_dqn = dqn_result["_agent"]
                    trained_dqn.epsilon = 0.0

                # Evaluate on original test cases
                correct_retrievals = 0
                correct_routings = 0
                silent_failures = 0
                correct_declines = 0
                total = 0
                decline_total = 0

                # One rule-based tool instance for baseline routing (avoids per-query instantiation)
                _rule_tool = _PKBTool(knowledge_base_path=sim.kb_path)

                for tc in self.test_cases:
                    features = sim.feature_extractor.extract(tc["query"])

                    # Routing
                    if trained_bandit:
                        arm = trained_bandit.select_arm(features)
                        strategy = ARM_NAMES[arm]
                    else:
                        # Baseline: rule-based keyword classifier (not oracle ground truth)
                        strategy = _rule_tool._classify_query(tc["query"])
                        arm = ARM_NAMES.index(strategy) if strategy in ARM_NAMES else 1

                    # Retrieval
                    cached = sim.cache.get(tc["query"], {})
                    search_results = cached.get(strategy, [])

                    # Check retrieval
                    expected_photo = tc.get("expected_photo")
                    if expected_photo:
                        photo_found = any(
                            expected_photo.lower() in r.get("photo_path", "").lower()
                            for r in search_results
                        )
                    else:
                        photo_found = True

                    if photo_found:
                        correct_retrievals += 1

                    # Routing accuracy
                    if strategy == tc.get("expected_type"):
                        correct_routings += 1

                    # Confidence decision
                    if trained_dqn:
                        state = ConfidenceState.from_retrieval(search_results, arm, features)
                        action = trained_dqn.select_action(state)
                        grade = action_to_grade(action)
                    else:
                        score = search_results[0]["relevance_score"] if search_results else 0.0
                        if score >= 0.7: grade = "A"
                        elif score >= 0.5: grade = "B"
                        elif score >= 0.35: grade = "C"
                        elif score >= 0.2: grade = "D"
                        else: grade = "F"

                    # Silent failure: confident but wrong
                    is_confident = grade in ("A", "B", "C")
                    if not photo_found and is_confident and expected_photo:
                        silent_failures += 1

                    # Decline accuracy
                    if tc.get("should_decline"):
                        decline_total += 1
                        if grade in ("D", "F"):
                            correct_declines += 1

                    total += 1

                config_results.append({
                    "seed": seed,
                    "retrieval_accuracy": correct_retrievals / total,
                    "routing_accuracy": correct_routings / total,
                    "silent_failure_rate": silent_failures / total,
                    "decline_accuracy": correct_declines / max(decline_total, 1),
                })

            # Aggregate across seeds
            metrics = {}
            for metric in ["retrieval_accuracy", "routing_accuracy",
                           "silent_failure_rate", "decline_accuracy"]:
                values = [r[metric] for r in config_results]
                metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }

            results[config_name] = {
                "config": config,
                "per_seed": config_results,
                "aggregated": metrics,
            }

            print(f"  retrieval={metrics['retrieval_accuracy']['mean']:.1%}, "
                  f"routing={metrics['routing_accuracy']['mean']:.1%}, "
                  f"silent_fail={metrics['silent_failure_rate']['mean']:.1%}, "
                  f"decline={metrics['decline_accuracy']['mean']:.1%}")

        # Save results
        results_path = "./eval/results/ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nAblation results saved to {results_path}")

        return results
