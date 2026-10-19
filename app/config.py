from datetime import date
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Data paths
    corpus_dir: str = "./data/corpus"
    manifest_path: str = "./data/manifest.json"
    db_path: str = "./data/veltrix.db"

    # Date override (YYYY-MM-DD). Empty = use system date.
    app_today: str = ""

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""

    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_today(self) -> date:
        if self.app_today:
            return date.fromisoformat(self.app_today)
        return date.today()


@lru_cache
def get_settings() -> Settings:
    return Settings()
