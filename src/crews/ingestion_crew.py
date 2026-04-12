"""Ingestion crew — processes photos into the knowledge base."""

from crewai import Crew, Process
from src.agents.definitions import create_photo_analyst, create_knowledge_retriever
from src.tasks.ingestion import create_scan_task, create_analyze_task, create_index_task


def create_ingestion_crew() -> Crew:
    """Create the sequential ingestion pipeline crew."""
    photo_analyst = create_photo_analyst()
    knowledge_retriever = create_knowledge_retriever()

    scan_task = create_scan_task(photo_analyst)
    analyze_task = create_analyze_task(photo_analyst, scan_task)
    index_task = create_index_task(knowledge_retriever, analyze_task)

    return Crew(
        agents=[photo_analyst, knowledge_retriever],
        tasks=[scan_task, analyze_task, index_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
        embedder={"provider": "sentence-transformer", "config": {"model": "all-MiniLM-L6-v2"}},
    )
