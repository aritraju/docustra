from fastapi import APIRouter, HTTPException

from docustra.api.schemas import QueryRequest, QueryResponse
from docustra.core import RetrievalError, get_logger
from docustra.retrieval import get_strategy

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info("Query received", pattern=request.pattern, question=request.question[:80])
    try:
        strategy = get_strategy(request.pattern)
        kwargs = {}
        if request.file_path:
            kwargs["file_path"] = request.file_path
        response = strategy.query(request.question, **kwargs)
        return QueryResponse.from_rag_response(response)
    except RetrievalError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Query failed", pattern=request.pattern, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.get("/patterns")
async def list_patterns() -> dict:
    from docustra.retrieval.base import RAGPattern
    return {
        "patterns": [
            {"id": p.value, "name": p.name.replace("_", " ").title()}
            for p in RAGPattern
        ]
    }
