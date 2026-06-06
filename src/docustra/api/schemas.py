from pydantic import BaseModel, Field

from docustra.retrieval.base import RAGPattern, RAGResponse


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    pattern: RAGPattern = RAGPattern.ADAPTIVE
    file_path: str | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What are the main risk factors disclosed in the 10-K filing?",
            "pattern": "corrective",
        }
    }}


class QueryResponse(BaseModel):
    answer: str
    pattern: str
    sources: list[dict]
    reasoning: str
    metadata: dict

    @classmethod
    def from_rag_response(cls, r: RAGResponse) -> "QueryResponse":
        return cls(
            answer=r.answer,
            pattern=r.pattern.value,
            sources=r.sources,
            reasoning=r.reasoning,
            metadata=r.metadata,
        )


class IngestRequest(BaseModel):
    file_path: str
    build_graph: bool = True


class IngestResponse(BaseModel):
    file: str
    chunks_indexed: int
    images_found: int
    graph_entities: int
    doc_ids: list[str]
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
