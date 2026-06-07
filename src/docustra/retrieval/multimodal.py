"""
Multimodal RAG
──────────────
Handles images (charts, diagrams) extracted from documents.
Uses Gemini Vision to generate text descriptions for each image at query time.
Image descriptions are retrieved alongside text chunks for a unified answer.
"""

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.ingestion.embedder import get_embeddings
from docustra.ingestion.parser import DocumentParser
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse, get_llm
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using both text and image description context.

Text Context:
{text_context}

Image Descriptions:
{image_context}""",
        ),
        ("human", "{question}"),
    ]
)


class MultimodalRAG(BaseRAGStrategy):
    pattern = RAGPattern.MULTIMODAL

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())
        self._parser = DocumentParser()
        # Vision-capable LLM (Gemini supports vision natively)
        self._vision_llm = get_llm()

    def query(self, question: str, file_path: str | None = None, **kwargs) -> RAGResponse:
        logger.info("Multimodal RAG query", question=question[:80])

        text_docs = self._vector_store.similarity_search(question)
        text_context = "\n\n".join(d.page_content for d in text_docs)

        image_descriptions = []
        if file_path and Path(file_path).exists():
            parsed = self._parser.parse(file_path)
            for img_data in parsed.images[:5]:
                desc = self._describe_image(img_data["b64"], img_data["ext"], question)
                if desc:
                    image_descriptions.append(f"[Image — Page {img_data['page']}]: {desc}")

        image_context = (
            "\n\n".join(image_descriptions) if image_descriptions else "No images available."
        )

        chain = _SYNTHESIS_PROMPT | self._llm
        answer = chain.invoke(
            {
                "question": question,
                "text_context": text_context,
                "image_context": image_context,
            }
        ).content  # type: ignore[union-attr]
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(text_docs),
            reasoning=f"Used {len(text_docs)} text chunks and {len(image_descriptions)} image descriptions.",
            metadata={"images_described": len(image_descriptions)},
        )

    def _describe_image(self, b64: str, ext: str, question: str) -> str:
        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Describe this image in detail, focusing on information relevant to: {question}",
                    },
                ]
            )
            response = self._vision_llm.invoke([message])
            return response.content  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Image description failed", error=str(e))
            return ""
