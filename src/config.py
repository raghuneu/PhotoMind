"""Configuration management with startup validation."""

from pydantic_settings import BaseSettings
from pydantic import Field


class PhotoMindSettings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str = Field(..., description="OpenAI API key for GPT-4o Vision")
    openai_model_name: str = Field(default="gpt-4o", description="Model for agent reasoning")
    serper_api_key: str = Field(default="", description="Optional SerperDev API key for web search")
    photos_directory: str = Field(default="./photos", description="Path to photo directory")
    knowledge_base_path: str = Field(
        default="./knowledge_base/photo_index.json",
        description="Path to the JSON knowledge base"
    )
    daily_budget_usd: float = Field(default=5.00, description="Daily API spend cap")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field(default="photos", description="Qdrant collection name")
    repository_backend: str = Field(
        default="json",
        description="Storage backend: 'json' (flat file) or 'qdrant' (vector DB)"
    )
    api_key: str = Field(
        default="",
        description="Optional API key for server auth. Empty string disables auth."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> PhotoMindSettings:
    """Load and validate settings. Fails fast if required keys are missing."""
    try:
        return PhotoMindSettings()
    except Exception as e:
        raise SystemExit(
            f"Configuration error: {e}\n"
            "Copy .env.example to .env and fill in your API keys."
        )


settings = get_settings()
