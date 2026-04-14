"""
PhotoMind - Personal Photo Knowledge Retrieval System

Usage:
    python -m src.main ingest              # Process photos into knowledge base
    python -m src.main query "question"    # Query the knowledge base
    python -m src.main eval               # Run evaluation suite
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "ingest":
        use_direct = "--direct" in sys.argv
        if use_direct:
            print("Running direct ingestion (bypasses CrewAI agents)...\n")
            from src.ingest_direct import run_direct_ingest
            run_direct_ingest()
        else:
            print("Running CrewAI ingestion crew...\n")
            try:
                from src.crews.ingestion_crew import create_ingestion_crew
                crew = create_ingestion_crew()
                result = crew.kickoff()
                print("\n" + "=" * 50)
                print("INGESTION COMPLETE")
                print("=" * 50)
                print(result)
            except Exception as e:
                print(f"\nCrewAI ingestion failed: {e}")
                print("Falling back to direct ingestion mode...\n")
                from src.ingest_direct import run_direct_ingest
                run_direct_ingest()

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python -m src.main query 'your question here'")
            return

        user_query = " ".join(sys.argv[2:])

        from src.crews.query_crew import create_query_crew

        crew = create_query_crew()
        result = crew.kickoff(inputs={"user_query": user_query})
        print("\n" + "=" * 50)
        print("PHOTOMIND ANSWER")
        print("=" * 50)
        print(result)

    elif command == "eval":
        from eval.run_evaluation import run_eval

        run_eval()

    elif command == "train":
        from src.rl.training_pipeline import TrainingPipeline
        from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS
        from eval.expanded_test_cases import ALL_TEST_CASES

        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRAINING_EPISODES
        pipeline = TrainingPipeline(test_cases=ALL_TEST_CASES)
        pipeline.train_full(n_episodes=episodes, seeds=SEEDS)

    elif command == "rl-eval":
        from eval.run_rl_evaluation import run_rl_eval

        run_rl_eval()

    elif command == "ablation":
        from src.rl.training_pipeline import TrainingPipeline
        from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS
        from eval.expanded_test_cases import ALL_TEST_CASES

        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRAINING_EPISODES
        pipeline = TrainingPipeline(test_cases=ALL_TEST_CASES)
        pipeline.run_ablation(n_episodes=episodes, seeds=SEEDS)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
