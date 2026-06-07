# Docustra — Architecture Deep Dive

## System Overview

Docustra is a unified document intelligence API where each query is handled by one of **9 RAG strategy implementations**. A single `POST /query` endpoint accepts a `pattern` parameter that selects the strategy at runtime.

**Production enhancements added:**
- **Hybrid RAG** — BM25 + dense vector retrieval fused with RRF, reranked by cross-encoder
- **Citation enforcement** — all patterns must cite sources or explicitly decline
- **Prompt versioning** — all LLM prompts stored in `prompts/v1/*.yaml`
- **CI/CD evaluation gate** — 50-pair golden dataset + RAGAS thresholds block regressions

---

## Core Data Flow

### Ingestion Pipeline

```
PDF File
  │
  ▼
DocumentParser (PyMuPDF)
  ├── Text blocks → List[Document]
  ├── Tables → List[Document]
  └── Images → List[{page, b64, ext}]
  │
  ▼
DocumentChunker (RecursiveCharacterTextSplitter)
  └── 650-token chunks, 100-token overlap  ← updated defaults (500-800 token sweet spot)
  │
  ├──▶ VectorStore.add_documents()  →  Qdrant collection
  │
  └──▶ KnowledgeGraphBuilder
          │
          ▼
        EntityExtractor (LLM)
          └── Entities + Relationships → Neo4j
```

### Query Pipeline

```
POST /query {question, pattern}
  │
  ▼
get_strategy(pattern)  →  Strategy instance
  │
  ▼
strategy.query(question)
  ├── VectorStore.similarity_search()
  ├── GraphStore.get_entity_context()   [Graph RAG only]
  ├── LLM calls (Gemini / Groq)
  └── RAGResponse {answer, sources, reasoning, metadata}
  │
  ▼
JSON response
```

---

## RAG Pattern Implementations

### 1. Adaptive RAG

**Problem solved:** Not every query needs deep retrieval — routing saves cost and latency.

```
Question
  │
  ▼
LLM Classifier → trivial | simple | complex
  │
  ├── trivial  → Direct LLM answer (no retrieval)
  ├── simple   → 1× vector search → answer
  └── complex  → Decompose → N× vector search → synthesize
```

**Key design:** The classifier is a single LLM call with a strict one-word response format, making it cheap (~50 tokens).

---

### 2. Agentic RAG

**Problem solved:** Multi-step queries that require iterative discovery.

```
HumanMessage (question)
  │
  ▼
LangGraph ReAct Loop
  ├── agent_node: LLM with bound tools
  │     Tools: [vector_search, web_search (optional)]
  ├── tools_node: ToolNode executes tool calls
  └── Loop until no tool_calls in AI response
  │
  ▼
Final AIMessage → answer
```

**Key design:** Uses LangGraph's `StateGraph` with `add_messages` reducer for conversation state. The agent self-terminates when it has sufficient context.

---

### 3. Branched RAG

**Problem solved:** Complex questions that span multiple independent sub-topics.

```
Question
  │
  ▼
LLM Decomposer → [sub_q1, sub_q2, sub_q3]
  │
  ▼
ThreadPoolExecutor (parallel)
  ├── Branch 1: vector_search(sub_q1) → answer_1
  ├── Branch 2: vector_search(sub_q2) → answer_2
  └── Branch 3: vector_search(sub_q3) → answer_3
  │
  ▼
LLM Synthesizer → final answer
```

**Key design:** Python `ThreadPoolExecutor` for true parallelism across branches. Results are deduped before synthesis.

---

### 4. Corrective RAG (CRAG)

**Problem solved:** Retrieved documents may be irrelevant, leading to hallucinated answers.

```
Question
  │
  ▼
VectorStore.similarity_search_with_scores()
  │
  ├── avg_score ≥ threshold (0.7) → use docs directly
  │
  └── avg_score < threshold
        │
        ├── LLM rewrites query → retry vector search
        │     ├── retry_score ≥ threshold → use retry docs
        │     └── retry_score < threshold → Tavily web search
        │
        ▼
       answer with whichever docs passed
```

**Key design:** Configurable `RETRIEVAL_SCORE_THRESHOLD` in `.env`. Tavily web search is optional — system degrades gracefully without it.

---

### 5. Graph RAG

**Problem solved:** Multi-hop questions about entity relationships that vector search misses.

```
Question
  │
  ├──▶ LLM Entity Extractor → [Apple, SEC, GDPR, ...]
  │
  ├──▶ VectorStore.similarity_search() → text_docs
  │
  └──▶ GraphStore.get_entity_context(entities)
         └── Neo4j Cypher: MATCH (e)-[r*1..2]-(neighbor)
  │
  ▼
LLM: answer using text_docs + graph_context
```

**Key design:** Graph traversal depth is configurable (default: 2 hops). Entity extraction runs on the query, not the answer, keeping it lightweight.

---

### 6. HyDE (Hypothetical Document Embedding)

**Problem solved:** Query embeddings look different from document embeddings — short queries vs. long paragraphs.

```
Short query: "Apple revenue drivers Q1 2024"
  │
  ▼
LLM generates hypothetical document:
  "Apple Inc. reported $94.9B revenue in Q1 2024,
   driven primarily by iPhone 15 series sales..."
  │
  ▼
embed(hypothetical_doc)  ← same space as real docs
  │
  ▼
VectorStore.similarity_search(hypothetical_doc)
  │
  ▼
LLM answers with real retrieved docs
```

**Key design:** The hypothetical document is NOT used as the answer — it's only the search vector. The final answer is generated from actually retrieved documents.

---

### 7. Multimodal RAG

**Problem solved:** Enterprise documents contain charts, diagrams, and tables that pure text RAG ignores.

```
PDF ingestion:
  ├── Text → VectorStore (as usual)
  └── Images → stored as base64 in memory (per-query processing)

Query time:
  ├── VectorStore.similarity_search() → text_docs
  └── For each image in target PDF:
        Gemini Vision: "Describe this image relevant to: {question}"
        → image_descriptions
  │
  ▼
LLM: answer from text_docs + image_descriptions
```

**Key design:** Images are described at query time (not ingestion) to keep descriptions contextually relevant to the specific question. Gemini 2.0 Flash supports vision natively.

---

### 8. Self-RAG

**Problem solved:** Silent hallucinations — the LLM generates answers that sound confident but aren't grounded.

```
Question
  │
  ▼
[Retrieve]: "Does this need retrieval?" → YES/NO
  │
  ├── NO → direct answer
  └── YES
        │
        ▼
      retrieve N docs
        │
        ▼
      [Relevant]: score each doc → YES/NO
        └── keep only relevant docs
        │
        ▼
      generate answer
        │
        ▼
      [Supported]: "Is answer grounded in context?" → YES/PARTIALLY/NO
        │
        ▼
      [Useful]: "Is this answer useful?" → YES/NO
        │
        ▼
      RAGResponse with full token audit log in `reasoning` field
```

**Key design:** All 4 reflection tokens are logged in `RAGResponse.reasoning` — making the system auditable. This is the pattern to use when answers will be reviewed by humans.

---

### 9. Hybrid RAG ⭐ NEW

**Problem solved:** Vector-only retrieval misses exact keyword matches; BM25-only misses semantic similarity. Neither enforces citation grounding.

```
Question
  │
  ├──────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      ▼
BM25 keyword search                        Dense vector search (Qdrant)
(rank-bm25 library)                        (sentence-transformers embedding)
  │                                                      │
  │  [rank 1: Doc-A, rank 2: Doc-C, ...]   [rank 1: Doc-B, rank 2: Doc-A, ...]
  │                                                      │
  └─────────────────────┬────────────────────────────────┘
                        │
                        ▼
          Reciprocal Rank Fusion (RRF)
          score = Σ weight / (60 + rank)
          BM25 weight=0.4, vector weight=0.6
                        │
                        ▼
          Cross-Encoder Reranker
          model: ms-marco-MiniLM-L-6-v2
          reads (query, doc) jointly → precise score
                        │
                        ▼
          Citation-Enforced LLM Answer
          (from prompts/v1/shared.yaml → citation_rag)
          Every claim must cite [Source: X, Page: Y]
          or model declines to answer
                        │
                        ▼
          RAGResponse
          ├── answer (with inline citations)
          ├── citations[] (structured list with reranker_score)
          └── metadata.declined (true if model couldn't answer)
```

**Key design:** Two-stage retrieval (fast candidate generation → precise reranking) achieves cross-encoder accuracy at bi-encoder speed. Citation enforcement is prompt-level — no post-processing required.

---

## Production Features

### Prompt Versioning

All LLM prompts are stored in YAML files under `prompts/<version>/`, not hardcoded in Python.

```
prompts/
└── v1/
    ├── shared.yaml        ← citation_rag, rag_basic (used by all patterns)
    ├── adaptive.yaml      ← router, direct_answer, decompose, synthesize
    ├── agentic.yaml       ← agent system message
    ├── branched.yaml      ← decompose, branch_answer, synthesize
    ├── corrective.yaml    ← relevance_score, rewrite_query
    ├── graph.yaml         ← entity_extract, graph_answer
    ├── hybrid.yaml        ← metadata/documentation
    ├── hyde.yaml          ← hypothetical_doc
    └── self_rag.yaml      ← retrieve_token, relevance_token, support_token, useful_token
```

The loader (`src/docustra/core/prompts.py`) reads the active version from `Settings.prompt_version` and caches results. Switching versions requires only a `.env` change — no code redeploy.

### Citation Enforcement

All retrieval strategies now:
1. Prepare context with `[Passage N | Source: X, Page: Y]` headers
2. Use `prompts/v1/shared.yaml → citation_rag` which mandates inline citations
3. Return a structured `citations[]` list alongside the answer text
4. Log `metadata.declined = true` when context is insufficient

The `RAGResponse` dataclass now includes:

```python
@dataclass
class RAGResponse:
    answer: str
    pattern: RAGPattern
    sources: list[dict]          # backward-compatible
    citations: list[dict]        # NEW: [{source, page, passage_preview, reranker_score}]
    reasoning: str
    metadata: dict               # includes prompt_version, declined
```

### CI/CD Evaluation Gate

```
Pull request changes retrieval or prompts
              │
              ▼
  .github/workflows/eval.yml triggers
              │
              ▼
  Qdrant starts as Docker service
              │
              ▼
  eval_ci.py runs 25 QA pairs from
  data/eval/golden_dataset.json
              │
              ▼
  RAGAS measures faithfulness,
  answer_relevancy, context_precision
              │
         ┌────▼────┐
         │ PASSES  │
         │ ≥ 0.70  │
         └────┬────┘
          YES │  NO
              │   │
            ✓ │   │ ✗
         Allow  Block
         merge   PR
```

---

## Storage Layer

### Qdrant (Vector Store)

- Collection: configurable via `QDRANT_COLLECTION`
- Distance: Cosine similarity
- Supports: similarity search, MMR (diversity), scored search
- Runs locally via Docker — no external API calls

### Neo4j (Graph Store)

- Entity types: COMPANY, PERSON, REGULATION, PRODUCT, LOCATION, CONCEPT
- Relationships: dynamically extracted by LLM, stored as Cypher relationship types
- APOC plugin enabled for advanced graph queries
- Neo4j Browser available at `http://localhost:7474`

---

## Configuration

All configuration is managed via Pydantic Settings (`src/docustra/core/config.py`).
Settings are loaded from `.env` with type validation and defaults.

Key tuning parameters:
- `RETRIEVAL_TOP_K` — number of documents retrieved per search
- `RETRIEVAL_SCORE_THRESHOLD` — CRAG confidence cutoff
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking strategy (defaults: 650 / 100)
- `BM25_WEIGHT` — weight for BM25 in RRF fusion (default: 0.4)
- `HYBRID_TOP_K` — candidates before reranking (default: 20)
- `ENABLE_RERANKING` — toggle cross-encoder reranking (default: true)
- `RERANKER_MODEL` — cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RERANKER_TOP_N` — final docs returned after reranking (default: 5)
- `PROMPT_VERSION` — active prompt version folder (default: `v1`)
- `EVAL_FAITHFULNESS_THRESHOLD` — CI gate threshold (default: 0.70)
- `EVAL_ANSWER_RELEVANCY_THRESHOLD` — CI gate threshold (default: 0.70)
- `EVAL_CONTEXT_PRECISION_THRESHOLD` — CI gate threshold (default: 0.60)
- `LLM_PROVIDER` — switch between Gemini and Groq without code changes

---

## Observability

Arize Phoenix provides local LLM tracing:
- Every LangChain / LangGraph call is auto-instrumented
- View trace trees, token counts, and latency at `http://localhost:6006`
- No data leaves your machine — fully local

---

## Evaluation

RAGAS metrics are implemented in `src/docustra/evaluation/metrics.py`:

| Metric | Measures | CI Threshold |
|---|---|---|
| Faithfulness | Are claims in the answer supported by retrieved context? | ≥ 0.70 |
| Answer Relevancy | Does the answer address the question asked? | ≥ 0.70 |
| Context Precision | Are the retrieved chunks actually used in the answer? | ≥ 0.60 |
| Context Recall | Does the retrieved context cover the ground truth answer? | Reported only |

### Golden Dataset

`data/eval/golden_dataset.json` — 50 QA pairs:
- **25 pairs** on Apple 10-K 2023 (financials, risk factors, products, legal)
- **25 pairs** on vector database concepts (indexing, search algorithms, RAG)

### Running evaluation

```bash
# Quick check (10 pairs)
uv run python scripts/eval_ci.py --sample 10 --pattern hybrid

# Full CI run (all 50 pairs)
uv run python scripts/eval_ci.py

# Compare patterns
for pattern in hybrid corrective adaptive; do
    uv run python scripts/eval_ci.py --pattern $pattern --sample 20 --output ${pattern}.json
done
```

### CI/CD integration

`.github/workflows/eval.yml` evaluates `hybrid`, `corrective`, and `adaptive` in parallel on every PR that touches retrieval, prompts, or evaluation files. Builds are blocked if any metric falls below threshold.
