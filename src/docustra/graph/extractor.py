"""Extracts entities and relationships from text using an LLM."""
import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.retrieval.base import get_llm

logger = get_logger(__name__)

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract entities and relationships from the text.
Return valid JSON with this exact structure:
{{
  "entities": [{{"name": "...", "type": "COMPANY|PERSON|REGULATION|PRODUCT|LOCATION|CONCEPT"}}],
  "relationships": [{{"from": "...", "type": "...", "to": "..."}}]
}}
Return ONLY the JSON. No markdown, no explanation.""",
        ),
        ("human", "{text}"),
    ]
)


class EntityExtractor:
    def __init__(self) -> None:
        self._llm = get_llm()

    def extract(self, text: str) -> dict:
        chain = _EXTRACTION_PROMPT | self._llm
        try:
            raw = chain.invoke({"text": text[:2000]}).content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Entity extraction failed", error=str(e), preview=text[:100])
            return {"entities": [], "relationships": []}

    def extract_from_documents(self, docs: list[Document]) -> list[dict]:
        results = []
        for doc in docs:
            extracted = self.extract(doc.page_content)
            extracted["source"] = doc.metadata.get("source", "unknown")
            results.append(extracted)
        return results
