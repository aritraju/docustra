class DocustraError(Exception):
    """Base exception for all Docustra errors."""


class IngestionError(DocustraError):
    """Raised when document ingestion fails."""


class RetrievalError(DocustraError):
    """Raised when retrieval fails across any pattern."""


class LLMError(DocustraError):
    """Raised when LLM call fails or returns unexpected output."""


class GraphError(DocustraError):
    """Raised when knowledge graph operations fail."""


class StorageError(DocustraError):
    """Raised when vector/graph store operations fail."""


class EvaluationError(DocustraError):
    """Raised when evaluation metrics computation fails."""
