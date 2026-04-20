"""Agent definitions for PhotoMind.

Communication Protocol
----------------------
Agents communicate through four mechanisms (see TECHNICAL_REPORT.md §6.4):

1. **Context passing** — CrewAI's ``context=[previous_task]`` passes the full
   output of one task as input context to the next.  In the ingestion crew,
   scan → analyze → index.  In the query crew, retrieval results (including
   relevance scores, evidence, and confidence grades) flow from the Knowledge
   Retriever's task to the Insight Synthesizer's task.

2. **Hierarchical manager delegation** — The Controller agent is the
   ``manager_agent`` in the query crew (``Process.hierarchical``).  It
   receives the user's query, formulates a plan (``planning=True``), and
   delegates sub-tasks to the Retriever and Synthesizer.

3. **Tool restriction as communication control** — The query task overrides
   the Retriever's tools to *only* ``PhotoKnowledgeBaseTool``, preventing
   fallback to ``JSONSearchTool``/``FileReadTool``.  This forces proper
   decline behaviour when no good match exists.

4. **Structured tool output as inter-agent schema** —
   ``PhotoKnowledgeBaseTool`` returns a JSON contract
   (``confidence_grade``, ``confidence_score``, ``source_photos``,
   ``warning``, ``answer_summary``) that the Synthesizer parses to produce
   its graded answer.  Grade F + warning → decline signal.
"""

from crewai import Agent
from crewai_tools import DirectoryReadTool, FileReadTool, JSONSearchTool

from src.tools.photo_knowledge_base import PhotoKnowledgeBaseTool
from src.tools.photo_vision import PhotoVisionTool
from src.config import settings

# SerperDevTool for web search enrichment — always imported, only active when key is set
from crewai_tools import SerperDevTool
try:
    serper_tool = SerperDevTool() if settings.serper_api_key else None
except Exception:
    serper_tool = None


def create_controller_agent() -> Agent:
    """Manager agent — classifies query intent and delegates to specialists.

    Decision-Making Protocol
    ------------------------
    The Controller implements a three-phase orchestration strategy:

    1. **Intent classification** — Before any delegation, the Controller
       analyzes the query to determine whether it requires factual entity
       extraction (amounts, dates, vendor names), semantic description
       matching (visual similarity, mood), or behavioral aggregation
       (frequency patterns across the corpus).  Classification cues are
       embedded in the agent's goal so that CrewAI's planning step
       (``planning=True``) generates the correct delegation plan.

    2. **Ambiguous query handling** — When a query mixes signals (e.g.,
       "Show me something I spent a lot on" combines semantic phrasing with
       factual intent), the Controller defers to the Retriever's
       ``PhotoKnowledgeBaseTool`` auto-classification, which may be
       overridden by the RL bandit.  The Controller's plan acknowledges
       ambiguity and instructs the Synthesizer to hedge confidence.

    3. **Delegation failure recovery** — If the Retriever returns grade F
       or an error payload, the Controller does NOT re-delegate to fallback
       tools.  Instead, it instructs the Synthesizer to produce a structured
       decline response.  If the Retriever task raises an exception (LLM
       timeout, malformed output), CrewAI's hierarchical manager catches the
       error and the Controller emits a grade-F decline with the error in
       the ``warning`` field.
    """
    return Agent(
        role="PhotoMind Controller",
        goal=(
            "Orchestrate photo knowledge retrieval by analyzing user queries, "
            "classifying intent (factual extraction, semantic search, or behavioral "
            "analysis), and delegating to the appropriate specialist agent. Always "
            "ensure answers include confidence scores and source photo attribution. "
            "IMPORTANT: Before delegating, classify the query intent and include "
            "the classification in your delegation plan. If the retrieval returns "
            "grade F or an error, instruct the Synthesizer to produce a decline "
            "response — never re-delegate to alternative tools."
        ),
        backstory=(
            "You are the chief librarian of a personal photo knowledge base. "
            "You understand that different questions need fundamentally different "
            "retrieval strategies. A question about a bill amount requires text "
            "extraction. A question about 'photos that feel like summer' requires "
            "semantic understanding. A question about 'what food do I photograph "
            "most' requires behavioral analysis. You classify first, then delegate. "
            "If you are unsure, you say so. If a subordinate agent fails or returns "
            "an error, you do not retry with different tools — you instruct the "
            "Synthesizer to report the failure honestly with confidence grade F."
        ),
        allow_delegation=True,
        verbose=True,
    )


def create_photo_analyst() -> Agent:
    """Processes and analyzes images — classification, OCR, entity extraction."""
    return Agent(
        role="Photo Analyst",
        goal=(
            "Analyze photos to extract structured knowledge: classify image type "
            "(bill, receipt, screenshot, food, scene, document, other), extract all "
            "visible text via OCR, generate a rich semantic description, and identify "
            "key entities (amounts, dates, vendor names, locations). Output a structured "
            "JSON knowledge record for each photo."
        ),
        backstory=(
            "You are a forensic image analyst specializing in extracting every piece "
            "of useful information from personal photographs. For a bill, you extract "
            "the exact dollar amount, vendor name, date, and account number. For a "
            "food photo, you describe the dish, cuisine type, setting, and mood. For "
            "a screenshot, you extract all visible text and identify the app or website. "
            "You never fabricate information not visible in the image. When text is "
            "unclear, you report your confidence level."
        ),
        tools=[
            PhotoVisionTool(),
            DirectoryReadTool(directory=settings.photos_directory),
        ],
        verbose=True,
    )


def create_knowledge_retriever() -> Agent:
    """Searches the knowledge base to answer queries.

    Design Note — Dual-Role Agent
    ------------------------------
    The Knowledge Retriever intentionally serves both the ingestion crew
    (writing the knowledge base) and the query crew (searching it).  This
    is a deliberate design choice, not under-specialization:

    1. **Shared knowledge base state** — A single agent holds the canonical
       understanding of the KB schema (``photo_index.json``).  Splitting
       into separate Indexer and Searcher agents would require both to
       agree on schema conventions, creating a synchronization risk.

    2. **Index consistency** — The agent that writes the KB is the same
       agent that reads it, eliminating format-mismatch bugs that arise
       when two agents independently interpret the same JSON structure.

    3. **Task-level tool restriction** — Although the agent is defined
       with three tools (PhotoKnowledgeBaseTool, FileReadTool, JSONSearchTool),
       the query task explicitly restricts available tools to only
       ``PhotoKnowledgeBaseTool``.  This means the agent's *behaviour*
       differs per crew despite being the same agent definition —
       a form of role polymorphism controlled by task configuration.
    """
    kb_path = settings.knowledge_base_path
    tools = [
        PhotoKnowledgeBaseTool(knowledge_base_path=kb_path),
        FileReadTool(),
        JSONSearchTool(json_path=kb_path),
    ]

    return Agent(
        role="Knowledge Retriever",
        goal=(
            "Search the photo knowledge base to find photos and extracted information "
            "that best answer the user's query. Use structured search for factual "
            "queries and semantic search for conceptual queries. Always return the "
            "source photo filename, a relevance score, and the specific evidence."
        ),
        backstory=(
            "You are a research librarian with perfect recall of every photo in the "
            "knowledge base. You understand that 'how much was my electric bill' "
            "requires searching extracted entities, while 'photos that remind me of "
            "vacation' requires matching against descriptions. You rank results by "
            "relevance and always cite your sources. When no good match exists, you "
            "say so clearly rather than returning a weak match disguised as confident."
        ),
        tools=tools,
        verbose=True,
    )


def create_insight_synthesizer() -> Agent:
    """Synthesizes answers with confidence grades and source attribution."""
    tools = [FileReadTool()]
    if serper_tool:
        tools.append(serper_tool)

    return Agent(
        role="Insight Synthesizer",
        goal=(
            "Synthesize retrieved photo evidence into clear, grounded answers with "
            "confidence scores (A-F) and source attribution. For behavioral queries, "
            "analyze patterns across multiple photos. Never fabricate details not "
            "present in the retrieved evidence."
        ),
        backstory=(
            "You are a senior analyst who turns raw retrieved evidence into "
            "trustworthy answers. Three rules: (1) Every claim must cite a specific "
            "source photo. (2) Every answer must include a confidence grade from A "
            "(high confidence) to F (no reliable match). (3) If the evidence is "
            "ambiguous or insufficient, you explicitly say so. You may use web search "
            "to enrich context, but the core answer always comes from the user's photos."
        ),
        tools=tools,
        verbose=True,
    )
