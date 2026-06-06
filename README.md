# Docustra — Enterprise Document Intelligence Platform

> A production-grade RAG system demonstrating all 8 advanced retrieval-augmented generation architectural patterns on real enterprise documents (SEC 10-K filings).

[![CI](https://github.com/aritraju/docustra/actions/workflows/ci.yml/badge.svg)](https://github.com/aritraju/docustra/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Project Demonstrates

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

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│  POST /query  →  RAG Strategy Router  →  Pattern Implementation  │
│  POST /ingest →  Parse → Chunk → Embed → Qdrant + Neo4j          │
└────────┬────────────────────────────┬────────────────────────────┘
         │                            │
    ┌────▼─────┐               ┌──────▼──────┐
    │  Qdrant  │               │   Neo4j     │
    │ (vectors)│               │ (KG entities│
    └──────────┘               │  & rels)    │
                               └─────────────┘
         │                            │
    ┌────▼────────────────────────────▼────┐
    │     Gemini 2.0 Flash / Groq LLM      │
    │     sentence-transformers embeds     │
    └──────────────────────────────────────┘
         │
    ┌────▼──────────────┐
    │  Streamlit UI      │
    │  Arize Phoenix     │
    │  (observability)   │
    └────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep-dive on each pattern's flow.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash (free API) · Groq Llama 3.3 70B (free API) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free, M2-optimized) |
| Vector Store | Qdrant (Docker) |
| Graph Store | Neo4j Community (Docker) |
| Orchestration | LangChain 0.3 · LangGraph |
| Web Search | Tavily API (free tier, CRAG fallback) |
| Backend API | FastAPI · Pydantic v2 |
| UI | Streamlit |
| Observability | Arize Phoenix (local tracing) |
| Document Parsing | PyMuPDF · Unstructured |
| Caching | Redis |
| Testing | pytest · pytest-asyncio · RAGAS |
| CI | GitHub Actions |
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

# Install dependencies
uv sync --extra dev --extra ui

# Copy environment config
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (free at aistudio.google.com)
```

### 3. Start Infrastructure

```bash
docker compose -f docker/docker-compose.yml up -d
```

This starts: Qdrant (`:6333`), Neo4j (`:7474`), Redis (`:6379`), Arize Phoenix (`:6006`)

### 4. Start the API

```bash
uv run docustra-api
# API runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 5. Ingest a Document

Download a sample SEC 10-K filing:
```bash
# Example: Apple 10-K (public domain)
python scripts/ingest_sample_data.py
```

Or via the API:
```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@apple_10k.pdf" \
  -F "build_graph=true"
```

### 6. Query with Any RAG Pattern

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main supply chain risk factors?",
    "pattern": "corrective"
  }'
```

### 7. Launch the UI

```bash
uv run streamlit run src/docustra/ui/app.py
```

---

## Running Tests

```bash
uv run pytest tests/unit/ -v
```

---

## Project Structure

```
docustra/
├── src/docustra/
│   ├── api/            # FastAPI routers (ingest, query, health)
│   ├── core/           # Config, logging, exceptions
│   ├── ingestion/      # Parse → Chunk → Embed pipeline
│   ├── retrieval/      # All 8 RAG pattern implementations
│   ├── graph/          # Neo4j entity extraction & KG builder
│   ├── storage/        # Qdrant + Neo4j abstractions
│   ├── evaluation/     # RAGAS metrics
│   └── ui/             # Streamlit demo app
├── tests/
│   ├── unit/           # Per-pattern unit tests
│   └── integration/    # End-to-end pipeline tests
├── docs/patterns/      # Deep-dive markdown per RAG pattern
├── docker/             # docker-compose.yml
├── scripts/            # Data ingestion helpers
└── notebooks/          # Pattern exploration notebooks
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check (Qdrant, Neo4j, Redis) |
| `/ingest` | POST | Ingest document by file path |
| `/ingest/upload` | POST | Upload and ingest a PDF |
| `/query` | POST | Query with any RAG pattern |
| `/query/patterns` | GET | List available RAG patterns |
| `/docs` | GET | Swagger UI |

---

## Observability

Every LangChain/LangGraph call is traced in **Arize Phoenix** at `http://localhost:6006`.
You can inspect the full retrieval chain, token usage, and latency for each pattern.

---

## License

MIT
