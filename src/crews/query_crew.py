"""Query crew — answers questions about the photo knowledge base.

Communication pattern: **Hierarchical delegation** (Process.hierarchical).
The Controller agent is the ``manager_agent``.  It receives the user query,
formulates a plan (``planning=True``), and delegates to specialist agents.

Data flow:
  1. Controller / manager   → classifies query intent, delegates
  2. Knowledge Retriever / query_task → searches KB via PhotoKnowledgeBaseTool
     (tools restricted at task level — JSONSearchTool/FileReadTool excluded)
     Returns structured JSON: confidence_grade, confidence_score,
     source_photos, warning, answer_summary
  3. Insight Synthesizer / synthesize_task (context=query_task)
     → graded answer with source attribution and confidence A–F

Tool restriction as communication control: The query_task overrides the
Retriever's tool list to *only* PhotoKnowledgeBaseTool.  This forces all
retrieval through the RL-enhanced pipeline and ensures proper decline
behaviour when no good match exists.

Memory: sentence-transformer embeddings (all-MiniLM-L6-v2) for cross-step
context retention.  The Controller can re-delegate if the initial result
is unsatisfactory (``allow_delegation=True``).
"""

from crewai import Crew, Process
from src.agents.definitions import (
    create_controller_agent,
    create_knowledge_retriever,
    create_insight_synthesizer,
)
from src.tasks.query import create_query_task, create_synthesize_task


def create_query_crew() -> Crew:
    """Create the hierarchical query pipeline crew.

    Error-Handling Strategy
    -----------------------
    CrewAI's hierarchical process provides top-level exception trapping:
    if a task raises, the manager agent receives the error and can
    re-plan.  The ``max_rpm`` parameter guards against OpenAI rate
    limits.

    If the crew fails, it returns a structured error dict with
    ``confidence_grade: "F"`` and the exception message in ``warning``,
    matching the decline contract expected downstream.
    """
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
        max_rpm=30,  # rate-limit guard for OpenAI API
    )
