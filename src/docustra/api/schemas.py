from pydantic import BaseModel, Field

from docustra.ingestion.chunker import ChunkingStrategy
from docustra.retrieval.base import RAGPattern, RAGResponse


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    pattern: RAGPattern = RAGPattern.ADAPTIVE
    file_path: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the main risk factors disclosed in the 10-K filing?",
                "pattern": "corrective",
            }
        }
    }


class QueryResponse(BaseModel):
    answer: str
    pattern: str
    sources: list[dict]
    citations: list[dict] = []
    reasoning: str
    metadata: dict

    @classmethod
    def from_rag_response(cls, r: RAGResponse) -> "QueryResponse":
        return cls(
            answer=r.answer,
            pattern=r.pattern.value,
            sources=r.sources,
            citations=r.citations,
            reasoning=r.reasoning,
            metadata=r.metadata,
        )


class IngestRequest(BaseModel):
    file_path: str
    build_graph: bool = False
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunking_params: dict = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "file_path": "/data/apple_10k_2023.pdf",
                "build_graph": False,
                "chunking_strategy": "parent_child",
                "chunking_params": {"parent_chunk_size": 1024, "child_chunk_size": 256},
            }
        }
    }


class IngestResponse(BaseModel):
    file: str
    chunks_indexed: int
    images_found: int
    graph_entities: int
    chunking_strategy: str = "recursive"
    doc_ids: list[str]
    error: str | None = None


class ChunkingParamSpec(BaseModel):
    """Describes one configurable parameter for a chunking strategy."""

    name: str
    label: str
    type: str  # "int" | "float" | "text" | "select"
    default: int | float | str
    options: list[str] | None = None  # only for type="select"
    min_val: int | float | None = None
    max_val: int | float | None = None
    help: str = ""


class ChunkingStrategyInfo(BaseModel):
    id: str
    name: str
    description: str
    requires_llm: bool
    best_for: str
    params: list[ChunkingParamSpec] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
