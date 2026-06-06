from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    google_api_key: str = ""
    groq_api_key: str = ""
    llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    llm_model: str = "gemini-2.0-flash"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "mps", "cuda"] = "cpu"

    # Vector Store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docustra_docs"

    # Graph Store
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "docustra_local"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Web Search
    tavily_api_key: str = ""

    # Observability
    phoenix_host: str = "localhost"
    phoenix_port: int = 6006
    enable_tracing: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    log_level: str = "INFO"

    # RAG
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 64

    ollama_base_url: str = "http://localhost:11434"

    @property
    def phoenix_endpoint(self) -> str:
        return f"http://{self.phoenix_host}:{self.phoenix_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
