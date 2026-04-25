"""
PhotoMind - Personal Photo Knowledge Retrieval System

Usage:
    python -m src.main ingest              # Process photos into knowledge base
    python -m src.main query "question"    # Query the knowledge base
    python -m src.main eval                # Run default eval suite (20 queries)
    python -m src.main eval --suite=expanded   # 56 queries
    python -m src.main eval --suite=held_out   # 14 held-out from train split
    python -m src.main eval --suite=novel      # 15 novel intent-shift queries
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
            print("Usage: python -m src.main query [--direct] 'your question here'")
            return

        use_direct = "--direct" in sys.argv
        args = [a for a in sys.argv[2:] if a != "--direct"]
        user_query = " ".join(args)

        if use_direct:
            # Zero-API-cost fast path: call the retrieval tool directly,
            # bypassing the CrewAI hierarchical pipeline (no GPT-4o calls).
            import json
            from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool

            repo = None
            try:
                from src.storage import get_repository
                repo = get_repository()
            except Exception:
                pass  # Fall back to JSON file

            tool = PhotoKnowledgeBaseTool(repository=repo)
            raw = tool._run(query=user_query, query_type="auto", top_k=5)
            parsed = json.loads(raw)
            print("\n" + "=" * 50)
            print("PHOTOMIND ANSWER (direct, no-API)")
            print("=" * 50)
            print(f"Query type: {parsed.get('query_type_detected', 'unknown')}")
            print(f"Confidence: {parsed.get('confidence_grade', 'F')} "
                  f"({parsed.get('confidence_score', 0.0)})")
            print(f"\n{parsed.get('answer_summary', '(no summary)')}")
            if parsed.get("source_photos"):
                print(f"\nSources ({len(parsed['source_photos'])}):")
                for p in parsed["source_photos"]:
                    print(f"  - {p}")
            if parsed.get("warning"):
                print(f"\n!!  {parsed['warning']}")
            return

        from src.crews.query_crew import create_query_crew

        crew = create_query_crew()
        result = crew.kickoff(inputs={"user_query": user_query})
        print("\n" + "=" * 50)
        print("PHOTOMIND ANSWER")
        print("=" * 50)
        print(result)

    elif command == "eval":
        from eval.run_evaluation import run_eval

        suite = "default"
        for arg in sys.argv[2:]:
            if arg.startswith("--suite="):
                suite = arg.split("=", 1)[1]
        run_eval(suite=suite)

    elif command == "train":
        from src.rl.training_pipeline import TrainingPipeline
        from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS
        from eval.expanded_test_cases import TRAIN_TEST_CASES

        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRAINING_EPISODES
        pipeline = TrainingPipeline(test_cases=TRAIN_TEST_CASES)
        pipeline.train_full(n_episodes=episodes, seeds=SEEDS)

    elif command == "rl-eval":
        from eval.run_rl_evaluation import run_rl_eval

        run_rl_eval()

    elif command == "ablation":
        from src.rl.training_pipeline import TrainingPipeline
        from src.rl.rl_config import N_TRAINING_EPISODES, SEEDS
        from eval.expanded_test_cases import TRAIN_TEST_CASES

        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else N_TRAINING_EPISODES
        pipeline = TrainingPipeline(test_cases=TRAIN_TEST_CASES)
        pipeline.run_ablation(n_episodes=episodes, seeds=SEEDS)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
