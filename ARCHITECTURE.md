# Docustra — Architecture Deep Dive

## System Overview

Docustra is a unified document intelligence API where each query is handled by one of 8 RAG strategy implementations. A single `POST /query` endpoint accepts a `pattern` parameter that selects the strategy at runtime.

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
  └── 512-token chunks, 64-token overlap
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
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking strategy
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

| Metric | Measures |
|---|---|
| Faithfulness | Are claims in the answer supported by retrieved context? |
| Answer Relevancy | Does the answer address the question asked? |
| Context Precision | Are the retrieved chunks actually used in the answer? |
| Context Recall | Does the retrieved context cover the ground truth answer? |
