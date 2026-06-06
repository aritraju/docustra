from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

import time
import unicodedata

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from docustra.core import get_settings


class RAGPattern(str, Enum):
    ADAPTIVE = "adaptive"
    AGENTIC = "agentic"
    BRANCHED = "branched"
    CORRECTIVE = "corrective"
    GRAPH = "graph"
    HYDE = "hyde"
    MULTIMODAL = "multimodal"
    SELF_RAG = "self_rag"


@dataclass
class RAGResponse:
    answer: str
    pattern: RAGPattern
    sources: list[dict] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Remove control characters that break JSON serialisation."""
    return "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C")
    )


def _normalise_content(msg: BaseMessage) -> BaseMessage:
    """Gemini 3.x returns content as a list of parts — flatten to clean plain string."""
    if isinstance(msg.content, list):
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in msg.content
        ).strip()
    else:
        text = str(msg.content)
    msg.content = _clean_text(text)
    return msg


class _NormalisedGemini(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI that always returns str content with clean characters."""

    def invoke(self, *args, **kwargs) -> BaseMessage:
        # Retry on 429 rate limit with exponential backoff
        for attempt in range(4):
            try:
                return _normalise_content(super().invoke(*args, **kwargs))
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise


class BaseRAGStrategy(ABC):
    """Base class for all RAG retrieval strategies."""

    pattern: RAGPattern

    def __init__(self) -> None:
        self._llm = get_llm()
        self._settings = get_settings()

    @abstractmethod
    def query(self, question: str, **kwargs) -> RAGResponse:
        ...

    def _format_sources(self, docs) -> list[dict]:
        return [
            {
                "content": d.page_content[:300],
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
            }
            for d in docs
        ]


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return _NormalisedGemini(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    return ChatGroq(
        model=settings.llm_model or "llama-3.3-70b-versatile",
        groq_api_key=settings.groq_api_key,
        temperature=0,
    )
