"""
PhotoVisionTool — image analysis via OpenAI GPT-4o Vision.

Extends crewai_tools.VisionTool with HEIC support via pillow-heif.
Converts any image (including HEIC) to JPEG in-memory, then sends
as base64 to the GPT-4o Vision API.

Inputs:
  - image_path (str): Path to image file (HEIC, PNG, JPG, WEBP supported)
  - analysis_prompt (str): Instructions for GPT-4o (defaults to structured extraction)

Outputs:
  - String with GPT-4o's analysis (image_type, ocr_text, description, entities, confidence)
"""

import base64
import io
import os
from typing import Type

import PIL.Image
import pillow_heif
from crewai.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

# Register HEIC/HEIF codec into PIL's opener registry once at import time.
# After this call, PIL.Image.open() transparently handles .HEIC files.
pillow_heif.register_heif_opener()

_DEFAULT_PROMPT = (
    "Analyze this image in detail. Provide a structured response with:\n"
    "1. image_type: classify as one of [bill, receipt, screenshot, food, "
    "scene, document, handwriting, other]\n"
    "2. ocr_text: transcribe ALL visible text verbatim (every word, number, date)\n"
    "3. description: a 2-3 sentence semantic description of the image\n"
    "4. entities: list every structured fact with type "
    "(amount, date, vendor, food_item, location, person, topic) and its value\n"
    "5. confidence: your extraction confidence from 0.0 to 1.0\n\n"
    "Be exhaustive with text extraction. For bills and receipts, always extract: "
    "total amount, vendor/company name, and date."
)


class PhotoVisionInput(BaseModel):
    image_path: str = Field(..., description="Path to the image file to analyze")
    analysis_prompt: str = Field(
        default=_DEFAULT_PROMPT,
        description="Instructions for how GPT-4o should analyze the image",
    )


class PhotoVisionTool(BaseTool):
    """Analyze images using GPT-4o Vision. Supports HEIC, PNG, JPG, WEBP.

    Error-handling contract
    ----------------------
    All failures are returned as structured error strings (never raised),
    so the calling agent always receives a parseable response.  Three
    failure classes are handled:

    * **Validation errors** (missing key, missing file) — instant return
      with a descriptive prefix ``"Error: ..."``.
    * **Transient API errors** (timeout, rate-limit 429, server 5xx) —
      retried up to ``_MAX_API_RETRIES`` times with exponential backoff.
    * **Permanent API errors** (auth 401, bad request 400) — returned
      immediately without retry.
    """

    name: str = "photo_vision"
    description: str = (
        "Analyze an image using GPT-4o Vision. Accepts HEIC, PNG, JPG, and WEBP files. "
        "Returns a structured analysis including image type, all visible text (OCR), "
        "semantic description, key entities (amounts, dates, vendors), and a confidence "
        "score. Use this for every photo during ingestion."
    )
    args_schema: Type[BaseModel] = PhotoVisionInput

    _MAX_API_RETRIES: int = 2
    _RETRY_BACKOFF_BASE: float = 1.0  # seconds

    @staticmethod
    def _handle_tool_error(image_path: str, error: Exception) -> str:
        """Standardized error formatting for all tool failures.

        Returns a structured error string that downstream agents can parse
        to distinguish validation errors from API errors.
        """
        error_type = type(error).__name__
        return f"Error [{error_type}] analyzing image {image_path}: {error}"

    def _run(self, image_path: str, analysis_prompt: str = _DEFAULT_PROMPT) -> str:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Error: OPENAI_API_KEY environment variable not set."

            if not os.path.exists(image_path):
                return f"Error: Image file not found at path: {image_path}"

            # Open image — pillow_heif handles HEIC transparently
            img = PIL.Image.open(image_path)
            # Normalize to RGB (HEIC may decode as RGBA or YCbCr)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Encode as JPEG in memory for the API call
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            client = OpenAI(api_key=api_key)

            # Retry loop for transient API errors (429 rate-limit, 5xx)
            import time
            last_error = None
            for attempt in range(self._MAX_API_RETRIES + 1):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            ],
                        }],
                        max_tokens=1000,
                        timeout=30.0,
                    )
                    return response.choices[0].message.content
                except Exception as api_err:
                    last_error = api_err
                    err_str = str(api_err).lower()
                    # Only retry on transient errors
                    is_transient = any(kw in err_str for kw in ["429", "rate", "timeout", "502", "503", "504"])
                    if is_transient and attempt < self._MAX_API_RETRIES:
                        time.sleep(self._RETRY_BACKOFF_BASE * (2 ** attempt))
                        continue
                    break

            return self._handle_tool_error(image_path, last_error)

        except FileNotFoundError:
            return f"Error: Image file not found at path: {image_path}"
        except Exception as e:
            return self._handle_tool_error(image_path, e)
