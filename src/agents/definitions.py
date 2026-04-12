"""Agent definitions for PhotoMind."""

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
    """Manager agent — classifies query intent and delegates to specialists."""
    return Agent(
        role="PhotoMind Controller",
        goal=(
            "Orchestrate photo knowledge retrieval by analyzing user queries, "
            "classifying intent (factual extraction, semantic search, or behavioral "
            "analysis), and delegating to the appropriate specialist agent. Always "
            "ensure answers include confidence scores and source photo attribution."
        ),
        backstory=(
            "You are the chief librarian of a personal photo knowledge base. "
            "You understand that different questions need fundamentally different "
            "retrieval strategies. A question about a bill amount requires text "
            "extraction. A question about 'photos that feel like summer' requires "
            "semantic understanding. A question about 'what food do I photograph "
            "most' requires behavioral analysis. You classify first, then delegate. "
            "If you are unsure, you say so."
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
    """Searches the knowledge base to answer queries."""
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
