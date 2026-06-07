from functools import lru_cache
from typing import Literal

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

    # RAG — chunk sizes follow 500-800 token / 100 overlap recommendation
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.7
    chunk_size: int = 650  # midpoint of 500-800 token target range
    chunk_overlap: int = 100  # preserves sentence context across boundaries

    # Hybrid retrieval (BM25 + vector)
    bm25_weight: float = 0.4  # weight for BM25 in RRF fusion (vector gets 1 - bm25_weight)
    hybrid_top_k: int = 20  # candidates fetched before reranking
    enable_reranking: bool = True  # toggle cross-encoder reranking

    # Cross-encoder reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free, ~80MB
    reranker_top_n: int = 5  # final docs returned after reranking

    # Prompt versioning
    prompt_version: str = "v1"

    # CI/CD evaluation thresholds — builds fail below these scores
    eval_faithfulness_threshold: float = 0.70
    eval_answer_relevancy_threshold: float = 0.70
    eval_context_precision_threshold: float = 0.60

    ollama_base_url: str = "http://localhost:11434"

    @property
    def phoenix_endpoint(self) -> str:
        return f"http://{self.phoenix_host}:{self.phoenix_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
