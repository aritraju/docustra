# Docustra — Enterprise Document Intelligence Platform

> A production-grade RAG system demonstrating **9 RAG architectural patterns**, **10 chunking strategies**, and **3 production reliability features** — all selectable at runtime via a modern three-tab Streamlit UI, backed by a FastAPI server, Qdrant vector store, and Neo4j knowledge graph.

[![CI](https://github.com/aritraju/docustra/actions/workflows/ci.yml/badge.svg)](https://github.com/aritraju/docustra/actions)
[![Eval Gate](https://github.com/aritraju/docustra/actions/workflows/eval.yml/badge.svg)](https://github.com/aritraju/docustra/actions/workflows/eval.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📸 **Screenshots** → [`docs/screenshots/`](docs/screenshots/README.md)

---

## What This Project Demonstrates

### 9 RAG Architectural Patterns

| Pattern | Description | When It Wins |
|---|---|---|
| **Adaptive RAG** | Routes queries by complexity — trivial/simple/complex | Saves cost on easy questions |
| **Agentic RAG** | LangGraph ReAct loop with tool use | Multi-step, open-ended queries |
| **Branched RAG** | Parallel sub-question retrieval + synthesis | Complex comparative questions |
| **Corrective RAG (CRAG)** | Scores retrieved docs; falls back to web search | Low-quality document coverage |
| **Graph RAG** | Neo4j knowledge graph + vector search | Multi-hop entity relationships |
| **HyDE** | Generates hypothetical doc → embeds it for search | Abstract / high-level queries |
| **Multimodal RAG** | Describes images/charts via Vision LLM | Image-rich annual reports |
| **Self-RAG** | LLM emits `[Retrieve]`, `[Relevant]`, `[Supported]` tokens | High-stakes, auditable answers |
| **⭐ Hybrid RAG** | BM25 + vector + RRF fusion + cross-encoder reranking | **Production "ask my docs"** |

### 3 Production Standards

| Feature | What It Does |
|---|---|
| **Citation Enforcement** | Every answer must cite `[Source: X, Page: Y]` — or explicitly decline. No hallucination. |
| **Prompt Versioning** | All LLM prompts stored in `prompts/v1/*.yaml` — tune without touching Python code. |
| **CI/CD Eval Gating** | 50-pair golden dataset + RAGAS metrics gate every pull request. Faithfulness < 0.70 blocks the build. |

### 10 Chunking Strategies (all runtime-selectable with configurable parameters)

| Strategy | Approach | LLM? |
|---|---|---|
| **Recursive** | Splits `\n\n` → `\n` → `.` → ` ` recursively (default) | No |
| **Character** | Single configurable separator | No |
| **Token** | tiktoken cl100k_base token count | No |
| **Sentence Transformers** | Embedding model's own tokeniser | No |
| **Semantic** | Topic-change detection via embedding similarity | No* |
| **Sentence Window** | Index sentences; retrieve ±N surrounding context | No |
| **Markdown** | Splits on `#`/`##`/`###` — header path in metadata | No |
| **HTML** | Splits on `<h1>`–`<h4>` — heading breadcrumb in metadata | No |
| **Parent-Child** | Small child indexed; large parent returned to LLM | No |
| **Hypothetical Questions** | LLM generates questions per chunk; questions embedded | Yes 🤖 |

*Uses the local embedding model, not the LLM API

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                            │
│  POST /query  →  RAG Strategy Router  →  Pattern Implementation   │
│  POST /ingest →  Parse → Chunk (650/100) → Embed → Qdrant+Neo4j  │
└────────┬──────────────────────────────┬───────────────────────────┘
         │                              │
    ┌────▼─────┐                  ┌─────▼──────┐
    │  Qdrant  │                  │   Neo4j    │
    │ (vectors)│                  │ (KG graph) │
    └──────────┘                  └────────────┘
         │
    ┌────▼──────────────────────────────────────┐
    │  Hybrid RAG pipeline (new)                 │
    │  BM25 ──┐                                  │
    │          ├─ RRF Fusion ─ Cross-Encoder ─▶  │
    │  Vector ─┘                                  │
    └────────────────────────────────────────────┘
         │
    ┌────▼──────────────────────────────────────┐
    │  Prompt Registry (prompts/v1/*.yaml)       │
    │  get_prompt("shared", "citation_rag")      │
    └────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────┐
    │  Gemini 2.0 Flash / Groq Llama 3.3 70B   │
    │  sentence-transformers/all-MiniLM-L6-v2   │
    └──────────────────────────────────────────┘
         │
    ┌────▼──────────────────┐   ┌───────────────────────┐
    │  Streamlit UI          │   │  CI/CD Eval Gate       │
    │  Citations panel       │   │  scripts/eval_ci.py    │
    │  Decline banner        │   │  50-pair golden dataset│
    │  Arize Phoenix traces  │   │  RAGAS thresholds      │
    └────────────────────────┘   └───────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep-dive on each pattern's flow.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash (free API) · Groq Llama 3.3 70B (free API) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | Qdrant (Docker) |
| Graph Store | Neo4j Community (Docker) |
| Orchestration | LangChain 0.3 · LangGraph |
| **Hybrid Retrieval** | **`rank-bm25` (BM25) · `sentence-transformers` CrossEncoder** |
| Web Search | Tavily API (free tier, CRAG fallback) |
| Backend API | FastAPI · Pydantic v2 |
| UI | Streamlit |
| Observability | Arize Phoenix (local tracing) |
| Document Parsing | PyMuPDF |
| Caching | Redis |
| **Evaluation** | **RAGAS · 50-pair golden dataset · `eval_ci.py`** |
| CI | GitHub Actions (with RAG eval gate) |
| Dependency Mgmt | `uv` |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker Desktop

### 2. Clone & Install

```bash
git clone https://github.com/aritraju/docustra.git
cd docustra

# Install all dependencies (including rank-bm25, sentence-transformers, ragas)
uv sync --extra dev --extra ui

# Copy environment config
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY (free at aistudio.google.com)
```

### 3. Start Infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

Starts: Qdrant (`:6333`), Neo4j (`:7474`), Redis (`:6379`), Arize Phoenix (`:6006`)

### 4. Start the API

```bash
uv run docustra-api
# API at http://localhost:8000 · Swagger at http://localhost:8000/docs
```

### 5. Ingest a Document

```bash
# Ingest sample data from data/ directory
python scripts/ingest_sample_data.py

# Or upload via API
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@data/apple_10k_2023.pdf" \
  -F "chunking_strategy=recursive"
```

### 6. Query with Hybrid RAG (recommended)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are Apples main supply chain risk factors?",
    "pattern": "hybrid"
  }'
```

**Response includes citations:**

```json
{
  "answer": "Apple relies on single-source suppliers [Source: apple_10k_2023.pdf, Page: 14] and manufacturing is concentrated in Asia [Source: apple_10k_2023.pdf, Page: 12].",
  "citations": [
    { "source": "apple_10k_2023.pdf", "page": 14, "reranker_score": 2.341 },
    { "source": "apple_10k_2023.pdf", "page": 12, "reranker_score": 1.892 }
  ],
  "metadata": { "declined": false, "prompt_version": "v1" }
}
```

### 7. Launch the UI

```bash
uv run streamlit run src/docustra/ui/app.py
# UI at http://localhost:8501
```

---

## Production Features

### Citation Enforcement

Every answer cites its sources or refuses to answer. Implemented via a versioned prompt in `prompts/v1/shared.yaml`:

```
"You are a precise document analysis assistant. Every factual claim must be
attributed using inline citations in the format [Source: <filename>, Page: <page>].
If the context does NOT contain sufficient information, respond with:
'I cannot answer this question based on the provided documents.'"
```

See [docs/production/01_citation_enforcement.md](docs/production/01_citation_enforcement.md)

### Prompt Versioning

All prompts live in `prompts/v1/*.yaml`. Create a new version by copying the folder:

```bash
cp -r prompts/v1 prompts/v2
# Edit prompts/v2/shared.yaml
# Set PROMPT_VERSION=v2 in .env
# No code changes needed
```

See [docs/production/02_prompt_versioning.md](docs/production/02_prompt_versioning.md)

### CI/CD Evaluation Gate

Run the evaluation locally:

```bash
# Quick check (10 pairs)
uv run python scripts/eval_ci.py --sample 10 --pattern hybrid

# Full evaluation (50 pairs) — same as CI
uv run python scripts/eval_ci.py

# Output:
# ✓ faithfulness:       0.8234  (threshold: 0.70)
# ✓ answer_relevancy:   0.7891  (threshold: 0.70)
# ✓ context_precision:  0.6723  (threshold: 0.60)
# ✓ All thresholds passed — build gates cleared.
```

GitHub Actions runs this automatically on every PR that touches retrieval or prompt files. See [docs/production/03_eval_ci_gating.md](docs/production/03_eval_ci_gating.md)

---

## Hybrid RAG Deep Dive

The new **Hybrid RAG** pattern is the recommended production choice. It combines:

1. **BM25 keyword search** — catches exact terms like "Section 12(g)" or "ASC 606"
2. **Dense vector search** — catches synonyms and paraphrases
3. **Reciprocal Rank Fusion** — merges both ranked lists; documents appearing in both rank highest
4. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores (query, doc) pairs jointly for precision

```
BM25 results:  [Doc-A(1), Doc-C(2), Doc-B(3)]
Vector results:[Doc-B(1), Doc-A(2), Doc-D(3)]
        │
        ▼ RRF fusion
Merged: [Doc-A, Doc-B, Doc-C, Doc-D]  ← both methods agreed on A and B
        │
        ▼ Cross-encoder reranking
Final:  [Doc-B(2.84), Doc-A(1.12), Doc-C(0.67)]  ← precise relevance scores
        │
        ▼ Citation-enforced LLM answer
```

Configurable via `.env`:

```env
BM25_WEIGHT=0.4        # keyword vs semantic balance (0=pure vector, 1=pure BM25)
HYBRID_TOP_K=20        # candidates before reranking
ENABLE_RERANKING=true  # toggle cross-encoder
RERANKER_TOP_N=5       # final docs returned
```

See [docs/patterns/09_hybrid_rag.md](docs/patterns/09_hybrid_rag.md)

---

## Running Tests

```bash
# Unit tests
uv run pytest tests/unit/ -v

# RAG evaluation (requires running Qdrant + ingested documents)
uv run python scripts/eval_ci.py --sample 10

# Validate all prompt YAML files
python3 -c "
import yaml, sys
from pathlib import Path
for f in Path('prompts').rglob('*.yaml'):
    yaml.safe_load(open(f))
    print(f'✓ {f}')
print('All prompts valid.')
"
```

---

## UI Overview

The Streamlit interface has three tabs:

| Tab | Purpose |
|---|---|
| **📄 Document Intelligence** | Upload PDFs · select chunking strategy · configure parameters |
| **🔍 RAG Query** | Ask questions · select any of 9 RAG patterns · view answer, citations, reasoning, and metadata |
| **📊 System Dashboard** | Service health (Qdrant, Neo4j, Redis) · vector count · links to Qdrant dashboard |

New in the **🔍 RAG Query** tab:
- **Citations panel** — every answer shows cited passages with reranker relevance scores
- **Decline banner** — yellow warning when the model refuses to answer due to insufficient context
- **Prompt version** shown in metadata for every response

---

## Project Structure

```
docustra/
├── src/docustra/
│   ├── api/            # FastAPI routers (ingest, query, health)
│   ├── core/
│   │   ├── config.py   # Settings with hybrid/reranking/eval/prompt params
│   │   ├── prompts.py  # Versioned prompt loader (NEW)
│   │   └── ...
│   ├── ingestion/      # Parse → Chunk (650/100 defaults) → Embed pipeline
│   ├── retrieval/
│   │   ├── hybrid.py   # BM25 + vector + RRF + reranking (NEW)
│   │   ├── reranker.py # CrossEncoderReranker (NEW)
│   │   └── ...         # All 9 RAG patterns (updated with citations)
│   ├── graph/          # Neo4j entity extraction & KG builder
│   ├── storage/        # Qdrant + Neo4j abstractions
│   ├── evaluation/     # RAGAS metrics with CI thresholds
│   └── ui/             # Streamlit (updated: citations panel, 9 patterns)
├── prompts/
│   └── v1/             # Versioned prompt YAML files (NEW)
│       ├── shared.yaml # citation_rag prompt (used by all patterns)
│       ├── adaptive.yaml
│       ├── corrective.yaml
│       └── ...
├── data/
│   ├── eval/
│   │   └── golden_dataset.json  # 50 QA pairs for CI evaluation (NEW)
│   └── *.pdf                    # Sample documents
├── scripts/
│   └── eval_ci.py      # CI evaluation gate script (NEW)
├── .github/
│   └── workflows/
│       └── eval.yml    # RAG evaluation GitHub Actions workflow (NEW)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── patterns/       # Per-pattern deep-dives (09_hybrid_rag.md added)
│   ├── chunking/       # Chunking strategy docs (updated defaults)
│   └── production/     # Production features docs (NEW folder)
│       ├── 00_overview.md
│       ├── 01_citation_enforcement.md
│       ├── 02_prompt_versioning.md
│       └── 03_eval_ci_gating.md
└── docker/
    └── docker-compose.yml
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check (Qdrant, Neo4j, Redis) |
| `/ingest` | POST | Ingest document by file path |
| `/ingest/upload` | POST | Upload and ingest a PDF |
| `/query` | POST | Query with any of 9 RAG patterns — returns `citations[]` |
| `/query/patterns` | GET | List all 9 available RAG patterns |
| `/docs` | GET | Swagger UI |

### Query response shape (all patterns)

```json
{
  "answer": "string — may contain inline [Source: X, Page: Y] citations",
  "pattern": "hybrid | adaptive | corrective | ...",
  "sources": [{"source": "...", "page": 1, "content": "..."}],
  "citations": [
    {
      "source": "apple_10k_2023.pdf",
      "page": 34,
      "passage_preview": "first 200 chars of the cited passage...",
      "reranker_score": 2.341
    }
  ],
  "reasoning": "explanation of what the pattern did",
  "metadata": {
    "prompt_version": "v1",
    "declined": false,
    "retrieval_method": "hybrid_bm25_vector_rrf"
  }
}
```

---

## Documentation

| Topic | Location |
|---|---|
| Architecture deep-dive | [ARCHITECTURE.md](ARCHITECTURE.md) |
| RAG patterns overview | [docs/patterns/00_overview.md](docs/patterns/00_overview.md) |
| Hybrid RAG (new) | [docs/patterns/09_hybrid_rag.md](docs/patterns/09_hybrid_rag.md) |
| Chunking strategies | [docs/chunking/00_overview.md](docs/chunking/00_overview.md) |
| **Citation enforcement** | [docs/production/01_citation_enforcement.md](docs/production/01_citation_enforcement.md) |
| **Prompt versioning** | [docs/production/02_prompt_versioning.md](docs/production/02_prompt_versioning.md) |
| **CI/CD eval gating** | [docs/production/03_eval_ci_gating.md](docs/production/03_eval_ci_gating.md) |

---

## Observability

Every LangChain/LangGraph call is traced in **Arize Phoenix** at `http://localhost:6006`.
Inspect the full retrieval chain, token usage, and latency for each pattern, including reranker scores.

---

## License

MIT
