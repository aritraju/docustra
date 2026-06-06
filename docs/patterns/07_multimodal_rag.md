# Multimodal RAG

## Overview

Multimodal RAG extends retrieval-augmented generation beyond text to handle **charts, diagrams, tables, and images** embedded in documents. Enterprise documents — especially annual reports, financial filings, and technical manuals — contain rich visual information that pure text RAG completely ignores.

Docustra's Multimodal RAG uses a **Vision Language Model (VLM)** to generate natural language descriptions of images at query time, then combines those descriptions with text retrieval to produce a unified, visually-informed answer.

---

## The Problem It Solves

A typical Apple 10-K contains:
- Revenue breakdown pie charts
- Multi-year revenue trend line charts
- Geographic distribution bar charts
- Segment performance comparison tables

A standard RAG pipeline over the PDF text would miss all of this. Questions like:
> "According to the chart, which product segment had the steepest revenue decline?"

...cannot be answered from text alone because the chart data isn't in the text.

---

## Architecture

### Ingestion Phase

```
PDF Document
     │
     ▼
┌─────────────────────────────────────┐
│         DocumentParser (PyMuPDF)     │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │  Text blocks │  │   Images     │  │
│  │  → Chunks   │  │   → base64   │  │
│  │  → Qdrant   │  │   (in memory)│  │
│  └─────────────┘  └──────────────┘  │
│                                     │
│  Note: Images stored as base64      │
│  in ParsedDocument object per query │
│  (not pre-indexed — described       │
│  at query time for relevance)       │
└─────────────────────────────────────┘
```

### Query Phase

```
     User Question + file_path
            │
     ┌──────┴──────┐
     │ (parallel)  │
     ▼             ▼
VectorStore     DocumentParser
.similarity     .parse(file_path)
_search()       → images list
     │             │
     │             ▼
     │      For each image:
     │      ┌────────────────────────────┐
     │      │  Gemini Vision LLM         │
     │      │                            │
     │      │  Input:                    │
     │      │  - base64 image            │
     │      │  - "Describe relevant to   │
     │      │    {question}"             │
     │      │                            │
     │      │  Output: text description  │
     │      │  e.g. "Bar chart showing   │
     │      │  iPhone revenue $200.6B,   │
     │      │  Services $85.2B..."       │
     │      └────────────────────────────┘
     │             │
     └──────┬──────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │         LLM Answer Generator         │
  │                                      │
  │  Context A: Retrieved text passages  │
  │  Context B: Image descriptions       │
  │  → Combined answer                   │
  └──────────────────────────────────────┘
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/multimodal.py`

### Image Extraction (Parser)

```python
# src/docustra/ingestion/parser.py
for page_num, page in enumerate(doc, start=1):
    for img_ref in page.get_images(full=True):
        xref = img_ref[0]
        base_image = doc.extract_image(xref)
        b64 = base64.b64encode(base_image["image"]).decode()
        result.images.append({
            "page": page_num,
            "b64": b64,
            "ext": base_image["ext"],  # "png", "jpeg", etc.
            "source": str(path),
        })
```

PyMuPDF extracts every embedded image from the PDF, converting it to base64 for transport to the Vision LLM.

### Vision Description

```python
def _describe_image(self, b64: str, ext: str, question: str) -> str:
    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{b64}"},
        },
        {
            "type": "text",
            "text": f"Describe this image in detail, focusing on information "
                    f"relevant to: {question}",
        },
    ])
    return self._vision_llm.invoke([message]).content
```

**Key design choice:** The question is embedded in the description prompt. This means the VLM focuses its description on aspects relevant to the query, rather than describing everything generically. A chart description for "What drove revenue growth?" will emphasize growth segments; for "What are the risks?" it will highlight declining or volatile metrics.

### Combined Synthesis

```python
_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question using both text and image description context.

Text Context:
{text_context}

Image Descriptions:
{image_context}"""),
    ("human", "{question}"),
])

answer = chain.invoke({
    "question": question,
    "text_context": text_context,
    "image_context": "\n\n".join(image_descriptions)
}).content
```

---

## Demo Examples

**Document corpus:** Apple 10-K 2023 (with embedded charts)

### Example 1 — Revenue Chart Analysis

**Request:**
```json
{
  "question": "According to the revenue breakdown charts, which product segment performed best in FY2023?",
  "pattern": "multimodal",
  "file_path": "data/apple_10k_2023.pdf"
}
```

**Image descriptions generated (for 3 charts):**

```
[Image — Page 31]: Bar chart showing Apple's net revenue by product and service 
for FY2023. iPhone: $200,583M (largest segment, ~52% of revenue, slight decline 
from FY2022's $205,489M). Services: $85,200M (growing, up from $78,129M, second 
largest). Mac: $29,357M (significant decline from $40,177M in FY2022). iPad: 
$28,300M (slight decline). Wearables/Home/Accessories: $39,845M (slight decline).
Services is the only segment showing year-over-year growth.

[Image — Page 32]: Pie chart showing revenue by geography. Americas: 42.9%, 
Europe: 24.6%, Greater China: 19.0%, Japan: 6.2%, Rest of Asia Pacific: 7.3%.

[Image — Page 33]: Line chart showing 5-year revenue trend (FY2019-2023). 
Revenue grew from $260B in FY2019 to peak $394B in FY2022, then declined to 
$383B in FY2023. First revenue decline in 5 years.
```

**Response:**
```json
{
  "answer": "Based on the revenue charts in the FY2023 10-K:\n\n**Services was the standout performer** — it was the only segment to show year-over-year revenue growth, increasing from $78.1B in FY2022 to $85.2B in FY2023 (+9.0%). This is particularly notable because it bucked the trend seen across all hardware segments:\n\n- iPhone: $200.6B (down from $205.5B, -2.4%)\n- Mac: $29.4B (down from $40.2B, -26.9%) — steepest decline\n- iPad: $28.3B (down slightly)\n- Wearables: $39.8B (down slightly)\n\n**Mac had the worst performance** with a 27% decline, attributed to post-COVID PC market normalization after the M-chip driven upgrade cycle of FY2021-2022.\n\nThe 5-year trend chart also reveals FY2023 was Apple's first year of total revenue decline since FY2019, making the Services growth even more strategically significant.",
  "pattern": "multimodal",
  "reasoning": "Used 5 text chunks and 3 image descriptions.",
  "metadata": {
    "images_described": 3
  }
}
```

---

### Example 2 — Geographic Distribution from Chart

**Request:**
```json
{
  "question": "What does the geographic revenue distribution chart show about Apple's market concentration risk?",
  "pattern": "multimodal",
  "file_path": "data/apple_10k_2023.pdf"
}
```

**VLM description of geographic chart:**
```
"Pie chart titled 'Net Revenue by Reportable Segment and Product Line.'
Americas segment dominates at 42.9% ($164.4B). Europe at 24.6% ($94.3B).
Greater China at 19.0% ($72.6B). Japan 6.2% ($23.9B). Rest of Asia Pacific 7.3%.
The Americas + Europe combination represents 67.5% of revenue, providing 
geographic diversification. However, Greater China's 19% represents significant 
single-country/region concentration given geopolitical tensions."
```

**Response:**
```json
{
  "answer": "The geographic distribution chart reveals a **moderately concentrated** revenue structure:\n\n**Concentration risk exists on two axes:**\n\n1. **Americas dominance (42.9%):** Over-reliance on the US market means Apple is highly sensitive to US consumer spending cycles and regulatory action (DOJ antitrust).\n\n2. **Greater China concentration (19.0%):** At $72.6B, Greater China is Apple's 3rd largest market and represents an asymmetric risk — the region could simultaneously disrupt both supply (manufacturing concentration) and demand (consumer boycotts or government restrictions). No other company of Apple's scale has this dual supply-demand concentration in a single geopolitically sensitive region.\n\n**Positive mitigant:** Europe (24.6%) and US (42.9%) together represent 67.5% of revenue from stable regulatory markets, providing a floor. However, the 10-K notes EU regulatory risk (DMA) as a growing concern for the European segment.",
  "pattern": "multimodal",
  "metadata": {
    "images_described": 2
  }
}
```

---

### Example 3 — Table Extraction

Even non-image tables in PDFs often render as images in some PDF generators. Multimodal RAG handles these:

**Request:**
```json
{
  "question": "What does the financial summary table show about Apple's key metrics over 5 years?",
  "pattern": "multimodal",
  "file_path": "data/apple_10k_2023.pdf"
}
```

**VLM description of financial table image:**
```
"Five-year financial summary table. Columns: FY2019, FY2020, FY2021, FY2022, FY2023.
Net revenue: $260.2B, $274.5B, $365.8B, $394.3B, $383.3B.
Net income: $55.3B, $57.4B, $94.7B, $99.8B, $97.0B.
Earnings per share (diluted): $11.97, $12.76, $5.67*, $6.15, $6.16.
Cash and equivalents: $48.8B, $38.0B, $34.9B, $23.6B, $29.9B.
(*adjusted for stock split)
Trend: Revenue peaked in FY2022, slight decline in FY2023.
Net income shows same peak-decline pattern. EPS relatively flat FY2022-2023."
```

---

## Limitations and Workarounds

### Limitation 1: Images processed at query time (not ingested)

Currently, Docustra processes images at query time. For large PDFs with many images, this adds latency.

**Production optimization:** Pre-process images at ingestion and store descriptions in Qdrant alongside text chunks:

```python
# Future enhancement (not yet implemented):
for img in parsed.images:
    desc = vision_llm.describe(img["b64"], "general document image")
    vector_store.add_documents([Document(
        page_content=desc,
        metadata={"source": path, "page": img["page"], "type": "image"}
    )])
```

### Limitation 2: 5-image cap per query

```python
for img_data in parsed.images[:5]:  # cap at 5 to control latency/cost
```

Increase this cap for image-dense documents.

### Limitation 3: Image quality

Very small images, watermarks, or decorative logos generate noisy descriptions. Add a size filter:

```python
if base_image["width"] < 100 or base_image["height"] < 100:
    continue  # skip tiny/decorative images
```

---

## Configuration

```env
LLM_PROVIDER=gemini   # Gemini 2.0 Flash supports vision natively
RETRIEVAL_TOP_K=5
```

**For image-heavy documents**, increase top_k slightly to ensure enough text context alongside image descriptions:
```env
RETRIEVAL_TOP_K=7
```

---

## When to Use Multimodal RAG

**Use when:**
- Documents contain charts, graphs, diagrams, or image-embedded tables
- Questions reference visual elements ("according to the chart...", "as shown in figure...")
- Annual reports, technical manuals, medical imaging reports, architectural drawings
- Benchmark reports where key findings are presented as infographics

**Avoid when:**
- Documents are pure text (adds latency with no benefit)
- Images are purely decorative (logos, backgrounds)
- File path is not available at query time

---

## Supported Image Types

| Format | Supported | Notes |
|---|---|---|
| JPEG/JPG | ✅ | Most common in PDFs |
| PNG | ✅ | Charts, diagrams |
| TIFF | ✅ via PyMuPDF | Scanned documents |
| SVG | ⚠️ | Converted to raster by PyMuPDF |
| Vector graphics | ⚠️ | Quality depends on rasterization |

---

## Related Patterns

- **Adaptive RAG** — add "multimodal" as a route option when file_path is provided
- **Self-RAG** — validate image description quality with [Relevant] reflection tokens
- **Graph RAG** — entities extracted from image descriptions can be added to the knowledge graph
