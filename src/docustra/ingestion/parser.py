import base64
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document


@dataclass
class ParsedDocument:
    text_chunks: list[Document] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)   # [{page, b64, caption}]
    tables: list[Document] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DocumentParser:
    """Parses PDFs into text, tables, and images for multimodal ingestion."""

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        result = ParsedDocument(metadata={"source": str(path), "filename": path.name})
        doc = fitz.open(str(path))

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                result.text_chunks.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "page": page_num, "type": "text"},
                    )
                )

            # Extract images for Multimodal RAG
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                base_image = doc.extract_image(xref)
                b64 = base64.b64encode(base_image["image"]).decode()
                result.images.append(
                    {
                        "page": page_num,
                        "index": img_index,
                        "b64": b64,
                        "ext": base_image["ext"],
                        "source": str(path),
                    }
                )

            # Extract tables via text blocks with tabular structure
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip()
                if block_text.count("\t") > 3 or block_text.count("  ") > 10:
                    result.tables.append(
                        Document(
                            page_content=block_text,
                            metadata={"source": str(path), "page": page_num, "type": "table"},
                        )
                    )

        doc.close()
        return result
