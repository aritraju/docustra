from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from docustra.core import get_logger, get_settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    logger.info("Loading embedding model", model=settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
