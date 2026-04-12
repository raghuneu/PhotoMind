"""Query crew — answers questions about the photo knowledge base."""

from crewai import Crew, Process
from src.agents.definitions import (
    create_controller_agent,
    create_knowledge_retriever,
    create_insight_synthesizer,
)
from src.tasks.query import create_query_task, create_synthesize_task


def create_query_crew() -> Crew:
    """Create the hierarchical query pipeline crew."""
    controller = create_controller_agent()
    retriever = create_knowledge_retriever()
    synthesizer = create_insight_synthesizer()

    query_task = create_query_task(retriever)
    synthesize_task = create_synthesize_task(synthesizer, query_task)

    return Crew(
        agents=[retriever, synthesizer],
        tasks=[query_task, synthesize_task],
        process=Process.hierarchical,
        manager_agent=controller,
        verbose=True,
        memory=True,
        embedder={"provider": "sentence-transformer", "config": {"model": "all-MiniLM-L6-v2"}},
        planning=True,
    )
