import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from docustra.api.schemas import IngestRequest, IngestResponse
from docustra.core import IngestionError, get_logger
from docustra.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)


@router.post("", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(request.file_path, build_graph=request.build_graph)
        return IngestResponse(**result)
    except IngestionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Unexpected ingestion error", error=str(e))
        raise HTTPException(status_code=500, detail="Ingestion failed unexpectedly.")


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    build_graph: bool = Form(default=True),
) -> IngestResponse:
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    pipeline = IngestionPipeline()
    try:
        result = pipeline.ingest(tmp_path, build_graph=build_graph)
        result["file"] = file.filename
        return IngestResponse(**result)
    except IngestionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
