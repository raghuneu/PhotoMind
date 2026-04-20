"""
Scaling benchmark for PhotoMind RL components.

Measures wall-clock time for key operations as a function of:
  - Corpus size (number of photos in knowledge base)
  - Query volume (number of queries per evaluation)
  - Training episodes

Produces a JSON report.

Usage:
    python scripts/scaling_benchmark.py
"""

import json
import os
import time
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.rl_config import (
    DQN_STATE_DIM, DQN_ACTION_DIM, N_CONTEXT_CLUSTERS,
    ARM_NAMES,
)
from src.rl.dqn_confidence import ConfidenceDQNAgent


def _random_state() -> np.ndarray:
    """Generate a random 8-dim state vector."""
    return np.random.rand(DQN_STATE_DIM).astype(np.float32)


def benchmark_dqn_inference(n_trials: int = 10000) -> dict:
    """Benchmark DQN forward pass latency."""
    agent = ConfidenceDQNAgent()
    agent.epsilon = 0.0  # pure greedy for consistent timing
    states = [_random_state() for _ in range(n_trials)]

    # Warm up
    for s in states[:100]:
        agent.select_action(s)

    start = time.perf_counter()
    for s in states:
        agent.select_action(s)
    elapsed = time.perf_counter() - start

    return {
        "operation": "DQN inference (forward pass)",
        "n_trials": n_trials,
        "total_seconds": round(elapsed, 4),
        "mean_ms": round(elapsed / n_trials * 1000, 4),
        "throughput_per_sec": round(n_trials / elapsed, 1),
    }


def benchmark_search_scaling(corpus_sizes: list[int] = None) -> list[dict]:
    """Benchmark keyword search at different corpus sizes."""
    if corpus_sizes is None:
        corpus_sizes = [25, 100, 500, 1000, 5000]

    results = []
    for n_photos in corpus_sizes:
        fake_kb = [
            {
                "description": f"A photo description for item {i} with some keywords food pizza",
                "ocr_text": f"OCR text content {i}" if i % 2 == 0 else "",
            }
            for i in range(n_photos)
        ]

        query_keywords = {"pizza", "receipt", "photo", "food", "restaurant"}
        n_queries = 100

        start = time.perf_counter()
        for _ in range(n_queries):
            for photo in fake_kb:
                desc_words = set(photo["description"].lower().split())
                ocr_words = set(photo.get("ocr_text", "").lower().split())
                all_words = desc_words | ocr_words
                overlap = len(query_keywords & all_words)
                _ = overlap / max(len(query_keywords), 1)
        elapsed = time.perf_counter() - start

        results.append({
            "corpus_size": n_photos,
            "n_queries": n_queries,
            "total_seconds": round(elapsed, 4),
            "mean_query_ms": round(elapsed / n_queries * 1000, 2),
            "complexity": "O(n_photos x avg_words_per_photo)",
        })

    return results


def benchmark_bandit_training(episode_counts: list[int] = None) -> list[dict]:
    """Benchmark bandit update time as a function of training episodes."""
    if episode_counts is None:
        episode_counts = [500, 1000, 2000, 5000, 10000]

    results = []
    for n_eps in episode_counts:
        alpha = np.ones((N_CONTEXT_CLUSTERS, len(ARM_NAMES)))
        beta = np.ones((N_CONTEXT_CLUSTERS, len(ARM_NAMES)))

        start = time.perf_counter()
        for _ in range(n_eps):
            cluster = np.random.randint(0, N_CONTEXT_CLUSTERS)
            samples = np.random.beta(alpha[cluster], beta[cluster])
            arm = np.argmax(samples)
            reward = float(np.random.rand() > 0.4)
            if reward > 0.5:
                alpha[cluster, arm] += 1
            else:
                beta[cluster, arm] += 1
        elapsed = time.perf_counter() - start

        results.append({
            "n_episodes": n_eps,
            "total_seconds": round(elapsed, 4),
            "mean_episode_ms": round(elapsed / n_eps * 1000, 4),
            "complexity": "O(1) per episode (Beta sampling + update)",
        })

    return results


def benchmark_dqn_training_step() -> dict:
    """Benchmark DQN training step (forward + backward)."""
    agent = ConfidenceDQNAgent()
    n_steps = 1000

    # Fill replay buffer
    for _ in range(200):
        state = _random_state()
        action = agent.select_action(state)
        reward = np.random.rand() * 2 - 0.5
        next_state = _random_state()
        agent.step(state, action, reward, next_state, done=True)

    # Benchmark training steps (force learning by calling _learn directly)
    start = time.perf_counter()
    for _ in range(n_steps):
        agent._learn()
    elapsed = time.perf_counter() - start

    return {
        "operation": "DQN training step (batch=64, forward+backward)",
        "n_steps": n_steps,
        "total_seconds": round(elapsed, 4),
        "mean_step_ms": round(elapsed / n_steps * 1000, 4),
        "complexity": "O(batch_size x network_params) per step",
    }


def run_all_benchmarks():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("PhotoMind Scaling Benchmark")
    print("=" * 60)

    results = {}

    print("\n1. DQN Inference Latency...")
    results["dqn_inference"] = benchmark_dqn_inference()
    print(f"   -> {results['dqn_inference']['mean_ms']:.3f} ms/query "
          f"({results['dqn_inference']['throughput_per_sec']:.0f} queries/sec)")

    print("\n2. Keyword Search Scaling (corpus size)...")
    results["search_scaling"] = benchmark_search_scaling()
    for r in results["search_scaling"]:
        print(f"   {r['corpus_size']:>5} photos -> {r['mean_query_ms']:.2f} ms/query")

    print("\n3. Bandit Training Scaling (episodes)...")
    results["bandit_training"] = benchmark_bandit_training()
    for r in results["bandit_training"]:
        print(f"   {r['n_episodes']:>5} episodes -> {r['total_seconds']:.3f}s total "
              f"({r['mean_episode_ms']:.4f} ms/ep)")

    print("\n4. DQN Training Step Latency...")
    results["dqn_training"] = benchmark_dqn_training_step()
    print(f"   -> {results['dqn_training']['mean_step_ms']:.3f} ms/step")

    # Summary
    print("\n" + "=" * 60)
    print("SCALING SUMMARY")
    print("=" * 60)
    print(f"DQN inference:       {results['dqn_inference']['mean_ms']:.3f} ms/query")
    print(f"Search (25 photos):  {results['search_scaling'][0]['mean_query_ms']:.2f} ms/query")
    print(f"Search (5K photos):  {results['search_scaling'][-1]['mean_query_ms']:.2f} ms/query")
    print(f"Bandit training:     {results['bandit_training'][2]['total_seconds']:.2f}s for 2000 eps")
    print(f"DQN training:        {results['dqn_training']['mean_step_ms']:.3f} ms/step")

    search_5k = results["search_scaling"][-1]["mean_query_ms"]
    search_25 = results["search_scaling"][0]["mean_query_ms"]
    growth = search_5k / max(search_25, 0.001)
    print(f"\nBottleneck: Keyword search scales linearly ({growth:.1f}x slower at 5K vs 25 photos)")
    print("Mitigation: Vector DB (ChromaDB/Qdrant) gives O(log n) retrieval")

    # Save
    os.makedirs("eval/results", exist_ok=True)
    out_path = "eval/results/scaling_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
