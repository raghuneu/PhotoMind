"""Task definitions for the query pipeline."""

from crewai import Task


def create_query_task(knowledge_retriever) -> Task:
    """Search the knowledge base for evidence relevant to the user's query."""
    return Task(
        description=(
            "Search the photo knowledge base to find evidence for this query:\n\n"
            "Query: {user_query}\n\n"
            "Steps:\n"
            "1. Use the photo_knowledge_base tool to search with query_type='auto'\n"
            "2. Review the returned results, confidence grade, and source photos\n"
            "3. Return ALL raw retrieval results including scores and evidence\n"
            "4. ALWAYS include the confidence_grade from the tool output\n"
            "5. ALWAYS include the specific source photo filenames\n\n"
            "IMPORTANT: Return the retrieval results faithfully. Do not fabricate "
            "information not found in the knowledge base."
        ),
        expected_output=(
            "The raw retrieval results including:\n"
            "- results: list of matching photos with relevance scores and evidence\n"
            "- confidence_grade: A/B/C/D/F from the search tool\n"
            "- source_photos: list of photo filenames\n"
            "- query_type: factual/semantic/behavioral (as detected by the tool)"
        ),
        agent=knowledge_retriever,
    )


def create_synthesize_task(insight_synthesizer, query_task) -> Task:
    """Synthesize retriever results into a graded, cited answer."""
    return Task(
        description=(
            "You are given the raw retrieval results from the Knowledge Retriever "
            "for this user query:\n\n"
            "Query: {user_query}\n\n"
            "Steps:\n"
            "1. Review the retriever's output: results, confidence scores, evidence\n"
            "2. Synthesize a clear, human-readable answer\n"
            "3. Assign a final confidence_grade (A, B, C, D, or F) based on "
            "evidence strength\n"
            "4. Cite every source photo filename explicitly\n"
            "5. If the evidence is insufficient (D/F grade), clearly state this "
            "and do NOT fabricate an answer\n"
            "6. For behavioral queries, summarize the pattern with statistics\n"
            "7. If available, use web search to add relevant context, but the "
            "core answer must come from the user's photos\n\n"
            "IMPORTANT: Format your final response as JSON with these exact keys:\n"
            '  "answer", "confidence_grade", "source_photos", "query_type", '
            '"reasoning", "warning"\n'
            "The confidence_grade MUST be exactly one of: A, B, C, D, F"
        ),
        expected_output=(
            "A JSON response with:\n"
            "- answer: plain-language answer to the query\n"
            "- confidence_grade: A/B/C/D/F\n"
            "- source_photos: list of photo filenames that support the answer\n"
            "- query_type: factual/semantic/behavioral\n"
            "- reasoning: brief explanation of how the answer was derived\n"
            "- warning: any caveats (or null if confident)"
        ),
        agent=insight_synthesizer,
        context=[query_task],
    )
