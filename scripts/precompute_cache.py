"""Pre-compute all search strategy results for offline training."""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.test_cases import TEST_CASES
from src.rl.simulation_env import PhotoMindSimulator


def main():
    kb_path = "./knowledge_base/photo_index.json"
    sim = PhotoMindSimulator(kb_path, TEST_CASES, augmentation_factor=1)

    print(f"Pre-computed {len(sim.cache)} queries x 3 strategies")
    print(f"Cache sample (first query):")
    first_query = TEST_CASES[0]["query"]
    for strategy, results in sim.cache[first_query].items():
        n = len(results)
        top = results[0]["relevance_score"] if results else 0
        print(f"  {strategy}: {n} results, top_score={top:.3f}")

    # Save cache
    cache_path = "./knowledge_base/strategy_cache.json"
    serializable = {}
    for query, strategies in sim.cache.items():
        serializable[query] = {}
        for s, results in strategies.items():
            serializable[query][s] = results
    with open(cache_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nCache saved to {cache_path}")


if __name__ == "__main__":
    main()
