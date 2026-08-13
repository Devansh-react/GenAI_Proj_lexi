"""Central configuration; model identity is configured once here."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    pinecone_api_key: str = ""
    pinecone_index_name: str = "docqa-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_revision: str = "main"
    embedding_dimension: int = 1024
    mistral_api_key: str = ""
    mistral_model: str = "ministral-8b-latest"
    retrieval_threshold: float = 0.62
    ledger_path: str = ".docqa_ledger.sqlite3"
    namespace: str = "legal-docs"
    max_attempts: int = 2
    context_token_budget: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
