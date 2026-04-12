"""Task definitions for the ingestion pipeline."""

from crewai import Task


def create_scan_task(photo_analyst) -> Task:
    """Scan the photo directory and list unprocessed images."""
    return Task(
        description=(
            "Scan the photo directory and list all image files "
            "(jpg, jpeg, png, heic, webp). For each file, record the filename, "
            "file path, and file size. Return a JSON array of file records. "
            "Skip any files that already exist in the knowledge base at "
            "'./knowledge_base/photo_index.json' (check by filename to ensure "
            "idempotent processing). If the knowledge base file does not exist, "
            "treat all photos as new."
        ),
        expected_output=(
            "A JSON array of objects, each with keys: filename, file_path. "
            "Only include files not already in the knowledge base."
        ),
        agent=photo_analyst,
    )


def create_analyze_task(photo_analyst, scan_task) -> Task:
    """Analyze each photo using GPT-4o Vision."""
    return Task(
        description=(
            "For each photo from the scan results, use the Vision tool to analyze "
            "the image. For EACH photo, extract:\n"
            "1. image_type: classify as one of [bill, receipt, screenshot, food, "
            "   scene, document, handwriting, other]\n"
            "2. ocr_text: ALL visible text in the image (every word, number, date)\n"
            "3. description: a 2-3 sentence semantic description of the image\n"
            "4. entities: a list of structured facts, each with 'type' "
            "   (amount, date, vendor, food_item, location, person, topic) and 'value'\n"
            "5. confidence: your confidence in the extraction (0.0 to 1.0)\n\n"
            "Return a JSON array with one record per photo. Be exhaustive with text "
            "extraction. For bills and receipts, always extract: total amount, "
            "vendor/company name, and date."
        ),
        expected_output=(
            "A JSON array where each element has keys: filename, file_path, "
            "image_type, ocr_text, description, entities (array of {type, value}), "
            "confidence."
        ),
        agent=photo_analyst,
        context=[scan_task],
    )


def create_index_task(knowledge_retriever, analyze_task) -> Task:
    """Build/update the JSON knowledge base from analyzed photos."""
    return Task(
        description=(
            "Take the analyzed photo records and build the knowledge base file at "
            "'./knowledge_base/photo_index.json'. Structure:\n"
            '{\n'
            '  "metadata": {"created_at": "...", "last_updated": "...", '
            '"total_photos": N},\n'
            '  "photos": [array of photo records]\n'
            '}\n\n'
            "Each photo record must have: id (generate a unique string), file_path, "
            "filename, image_type, ocr_text, description, entities (array), "
            "confidence, indexed_at (current timestamp). "
            "If the knowledge base already exists, load it and append only new photos. "
            "Write the updated JSON to the file. Report how many photos were added."
        ),
        expected_output=(
            "Confirmation: 'Knowledge base updated: X new photos added. "
            "Total photos indexed: Y.' The full JSON must be written to the file."
        ),
        agent=knowledge_retriever,
        context=[analyze_task],
        output_file="./knowledge_base/photo_index.json",
    )
