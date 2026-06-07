# Hybrid RAG — BM25 + Vector Search with Cross-Encoder Reranking

**Pattern ID:** `hybrid` | **New in:** Production Enhancement Sprint | **Citation-Enforced:** Yes

---

## Overview

Hybrid RAG is a **production-grade retrieval pipeline** that combines the strengths of two fundamentally different search methods, then applies a precision layer on top:

1. **BM25 keyword search** — catches exact terms, rare words, and proper nouns
2. **Dense vector (semantic) search** — catches meaning, synonyms, and paraphrases
3. **Reciprocal Rank Fusion (RRF)** — merges both ranked lists intelligently
4. **Cross-encoder reranking** — re-scores the merged candidates with a precision model
5. **Citation enforcement** — the final answer must cite every claim or decline to answer

This is the pattern you should use when answer quality matters most and you need provable, auditable results.

---

## The Problem Each Layer Solves

### Why pure vector search isn't enough

Imagine you're searching a legal contract for the clause **"Section 12(g) of the Securities Exchange Act"**.

A user asks: *"What does Section 12(g) say?"*

- The **vector embedding** of `"Section 12(g)"` looks similar to embeddings of completely unrelated sentences about "sections" or "acts" — the exact term gets diluted.
- **BM25** nails it immediately because it's built for exact keyword matching.

Conversely, if a user asks *"What are the main ownership risks?"* and the document says *"shareholding concentration hazards"*, BM25 finds nothing (different words), but vector search succeeds (same meaning).

**You need both.**

### Why the top vector/BM25 results still need reranking

First-stage retrievers — both BM25 and bi-encoder vector search — encode the query and documents **independently**. They can't compare them together. A cross-encoder reads the (query, document) pair jointly, like a human would, and produces a much more accurate relevance score.

Think of it this way:
- **Bi-encoder (vector search):** *"Is this document generally about the same topic?"*
- **Cross-encoder (reranker):** *"Does this specific passage directly answer this specific question?"*

---

## Architecture

```
User Question
      │
      ├─────────────────────────────────────────────────────────┐
      │                                                         │
      ▼                                                         ▼
┌─────────────────────────┐                   ┌────────────────────────────┐
│   BM25 Keyword Search    │                   │  Dense Vector Search        │
│   (rank-bm25 library)    │                   │  (Qdrant cosine similarity) │
│                          │                   │                            │
│  Splits query into words │                   │  Embeds query as 384-dim   │
│  Scores docs by TF-IDF   │                   │  vector, finds nearest     │
│  Returns top-K ranked    │                   │  vectors in collection     │
└─────────────────────────┘                   └────────────────────────────┘
      │                                                         │
      │  [Doc-A rank 1, Doc-C rank 2, Doc-B rank 3 ...]        │
      │                                                         │
      │  [Doc-B rank 1, Doc-A rank 2, Doc-D rank 3 ...]        │
      │                                                         │
      └────────────────────┬────────────────────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │  Reciprocal Rank Fusion  │
             │                         │
             │  RRF score per doc =    │
             │  Σ weight / (60 + rank) │
             │                         │
             │  Merges both lists into │
             │  one ranked list        │
             └────────────┬────────────┘
                          │
                          │  [Doc-A(0.023), Doc-B(0.021), Doc-C(0.018)...]
                          │
                          ▼
             ┌─────────────────────────┐
             │  Cross-Encoder Reranker  │
             │                         │
             │  Takes (query, doc) pair │
             │  jointly — reads both   │
             │  at once                │
             │                         │
             │  Model: ms-marco-       │
             │  MiniLM-L-6-v2          │
             │  (free, ~80MB)          │
             └────────────┬────────────┘
                          │
                          │  [Doc-B(1.84), Doc-A(1.12), Doc-C(0.67)...]
                          │
                          ▼
             ┌─────────────────────────┐
             │  Citation-Enforced LLM   │
             │  Answer Generation       │
             │                         │
             │  Must cite [Source: X,  │
             │  Page: Y] for every     │
             │  claim, or decline      │
             └────────────┬────────────┘
                          │
                          ▼
                    RAGResponse
                    + citations[]
                    + reranker_scores
```

---

## Concept Deep-Dive: BM25

**BM25 (Best Match 25)** is the backbone of traditional search engines (it powers Elasticsearch and Solr by default).

### How it works

BM25 assigns a relevance score to each document based on:
1. **Term frequency (TF):** How many times does the query word appear in the document? More is better, but with diminishing returns (so a word appearing 100× isn't 100× better than 1×).
2. **Inverse document frequency (IDF):** How rare is this word across all documents? Rare words like "EBITDA" carry more signal than common words like "the".
3. **Document length normalization:** A short document that mentions a term is probably more focused on it than a long document.

### Example

```
Query: "Section 12g securities exchange act"

Documents in corpus:
  Doc-A: "...compliance with Section 12(g) of the Securities Exchange Act of 1934..."
  Doc-B: "...annual report filing requirements under federal securities law..."
  Doc-C: "...risk factors include competition, regulatory changes, and market volatility..."

BM25 scores:
  Doc-A: 18.4  ← "Section", "12g", "securities", "exchange", "act" all present → HIGH
  Doc-B:  3.2  ← only "securities" matches → LOW
  Doc-C:  0.8  ← no match → VERY LOW
```

BM25 would return Doc-A at rank 1. Vector search might return Doc-B at rank 1 (because it's semantically about financial filings). **RRF combines both** so Doc-A scores highest overall.

### In code
```python
from rank_bm25 import BM25Okapi

corpus = [doc.page_content.lower().split() for doc in all_docs]
bm25 = BM25Okapi(corpus)

query_tokens = "Section 12g securities exchange act".lower().split()
scores = bm25.get_scores(query_tokens)
# scores[i] = BM25 relevance of doc i for this query
```

---

## Concept Deep-Dive: Reciprocal Rank Fusion

**RRF** is a rank combination algorithm. It answers the question: *"If both BM25 and vector search agree a document is good, rank it highest. If only one likes it, give partial credit."*

### The formula

```
RRF score = Σ (weight / (k + rank))

where:
  k = 60  (smoothing constant — standard value from original RRF paper)
  rank = position in the ranked list (1 = best)
  weight = how much to trust each list (BM25=0.4, vector=0.6 by default)
```

### Example walkthrough

```
BM25 results (weight = 0.4):         Vector results (weight = 0.6):
  Rank 1: Doc-A                         Rank 1: Doc-B
  Rank 2: Doc-C                         Rank 2: Doc-A
  Rank 3: Doc-B                         Rank 3: Doc-D

RRF scores:
  Doc-A: 0.4/(60+1) + 0.6/(60+2) = 0.00656 + 0.00968 = 0.01624
  Doc-B: 0.4/(60+3) + 0.6/(60+1) = 0.00635 + 0.00984 = 0.01619
  Doc-C: 0.4/(60+2) + 0.0        = 0.00645
  Doc-D: 0.0        + 0.6/(60+3) = 0.00952

Final ranking: Doc-A (0.0162) > Doc-B (0.0162) > Doc-D (0.0095) > Doc-C (0.0065)
```

Notice how Doc-D (only from vector) beats Doc-C (only from BM25) because vector search has a higher weight. You can tune `bm25_weight` in `.env`.

### Why k=60?

The constant `k=60` prevents a document ranked #1 from getting an infinitely higher score than #2. It also means the difference between rank 1 and rank 10 is significant, but between rank 50 and rank 60 is negligible — which is exactly what we want.

---

## Concept Deep-Dive: Cross-Encoder Reranking

This is where precision is won or lost.

### Bi-encoder vs cross-encoder (the core difference)

```
BI-ENCODER (what vector search uses):
─────────────────────────────────────
  Query ──► Encoder ──► vector Q
  Doc   ──► Encoder ──► vector D
  
  similarity = cosine(Q, D)
  
  ✓ Fast: encode once, reuse embeddings
  ✗ Limited: can't compare Q and D together


CROSS-ENCODER (what the reranker uses):
─────────────────────────────────────────
  [Query + Doc] ──► Encoder ──► relevance score
  
  The model READS BOTH simultaneously.
  It can see: "this specific word in the doc 
              directly answers that specific 
              part of the query"
  
  ✗ Slow: must run for every (query, doc) pair
  ✓ Accurate: ~15-20% higher recall@10 vs bi-encoder
```

### Why not use cross-encoder for everything?

If you have 10,000 documents and must run a cross-encoder on every one, that's 10,000 model inferences per query — impossibly slow.

The solution is the **two-stage pipeline:**
1. **Stage 1 (fast):** BM25 + vector search → get top 20 candidates quickly
2. **Stage 2 (precise):** Cross-encoder → rescore only those 20 → return top 5

This gives you near-cross-encoder accuracy at near-bi-encoder speed.

### Our model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

- **Trained on:** MS MARCO passage ranking dataset (530,000 queries + Microsoft Bing clicks)
- **Size:** ~80MB — downloads once, runs locally
- **Speed:** ~50ms for 20 documents on CPU
- **Free:** Yes, open source on HuggingFace

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
    ("What are Apple's risk factors?", "Apple faces competition from Samsung..."),
    ("What are Apple's risk factors?", "Apple's revenue grew 5% in Q3..."),
]

scores = model.predict(pairs)
# scores = [2.41, -0.87]
# Higher is more relevant. Doc 1 directly answers the question.
```

---

## Citation Enforcement

Every answer from Hybrid RAG must include inline citations. This is enforced at the prompt level.

### How the context is prepared

Before passing documents to the LLM, each passage is wrapped with its source metadata:

```
[Passage 1 | Source: apple_10k_2023.pdf, Page: 23]  relevance: 2.341
Apple Inc. is exposed to market risk related to changes in interest rates...

[Passage 2 | Source: apple_10k_2023.pdf, Page: 45]  relevance: 1.892
The Company faces competition in all its markets, including from companies
with greater resources...
```

### The citation prompt (from `prompts/v1/shared.yaml`)

The LLM is given strict instructions:

```yaml
citation_rag:
  system: |
    You are a precise document analysis assistant. Every factual claim must be
    attributed using inline citations in the format [Source: <filename>, Page: <page>].
    
    If the context does NOT contain sufficient information, respond with:
    "I cannot answer this question based on the provided documents."
    Do NOT speculate or use outside knowledge.
```

### Example answer with citations

```
Question: "What are Apple's main supply chain risks?"

Answer: Apple's supply chain is concentrated in Asia, particularly China, which
exposes it to geopolitical risks and potential manufacturing disruptions
[Source: apple_10k_2023.pdf, Page: 12]. The company relies on single-source
suppliers for certain components, creating concentration risk if a supplier
cannot deliver [Source: apple_10k_2023.pdf, Page: 14]. Additionally, global
logistics disruptions could adversely affect product availability and
financial results [Source: apple_10k_2023.pdf, Page: 15].
```

### When the model declines

```
Question: "What is Apple's secret formula for iPhone chips?"

Answer: I cannot answer this question based on the provided documents.
The retrieved context does not contain sufficient information.
```

The UI shows a **yellow warning banner** when this happens, so users know to rephrase or ingest more documents.

---

## Demo

### 1. Start the API

```bash
uv run docustra-api
```

### 2. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@data/apple_10k_2023.pdf" \
  -F "chunking_strategy=recursive"
```

### 3. Query with Hybrid RAG

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are Apples main risk factors related to supply chain?",
    "pattern": "hybrid"
  }'
```

### 4. Example response

```json
{
  "answer": "Apple's supply chain risks include reliance on single-source suppliers [Source: apple_10k_2023.pdf, Page: 14] and manufacturing concentration in Asia [Source: apple_10k_2023.pdf, Page: 12].",
  "pattern": "hybrid",
  "sources": [
    {"source": "apple_10k_2023.pdf", "page": 14, "content": "The Company relies on single-source..."},
    {"source": "apple_10k_2023.pdf", "page": 12, "content": "Manufacturing is concentrated..."}
  ],
  "citations": [
    {
      "source": "apple_10k_2023.pdf",
      "page": 14,
      "passage_preview": "The Company relies on single-source suppliers...",
      "reranker_score": 2.341
    },
    {
      "source": "apple_10k_2023.pdf",
      "page": 12,
      "passage_preview": "Manufacturing is concentrated in Asia...",
      "reranker_score": 1.892
    }
  ],
  "reasoning": "Hybrid retrieval: BM25 weight=0.4, vector weight=0.6, reranked=True, final_docs=5",
  "metadata": {
    "retrieval_method": "hybrid_bm25_vector_rrf",
    "reranking": true,
    "bm25_weight": 0.4,
    "docs_retrieved": 5,
    "prompt_version": "v1",
    "declined": false
  }
}
```

### 5. Python SDK example

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What did Apple spend on R&D in fiscal 2023?",
    "pattern": "hybrid"
})
data = response.json()

print("Answer:", data["answer"])
print()
print("Citations:")
for i, c in enumerate(data["citations"], 1):
    print(f"  [{i}] {c['source']}, Page {c['page']} (relevance: {c['reranker_score']:.3f})")
    print(f"       {c['passage_preview'][:100]}...")
```

---

## Configuration

All hybrid parameters are configurable via `.env`:

```env
# Fusion weights (must sum to 1.0 approximately)
BM25_WEIGHT=0.4          # how much to trust BM25 keyword results
                         # vector gets 1 - BM25_WEIGHT = 0.6

# Candidate pool before reranking
HYBRID_TOP_K=20          # how many docs to fetch from each method

# Cross-encoder reranker
ENABLE_RERANKING=true    # set to false for faster (less accurate) mode
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=5         # final docs returned after reranking

# Prompt version
PROMPT_VERSION=v1
```

### Tuning the BM25 weight

| `BM25_WEIGHT` | When to use |
|---|---|
| `0.2` | Documents with rich semantic structure, casual language |
| `0.4` | General purpose (default) — balanced |
| `0.6` | Technical/legal documents with precise terminology |
| `0.8` | Codebases, regulatory text where exact terms are critical |

### Disabling reranking (faster mode)

```env
ENABLE_RERANKING=false
HYBRID_TOP_K=10
```

With reranking disabled, the pipeline returns the top-N from RRF fusion directly. About 2x faster but ~10-15% lower precision.

---

## Performance Characteristics

| Metric | Pure Vector | Pure BM25 | Hybrid+Rerank |
|---|---|---|---|
| Semantic similarity | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Exact term matching | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Recall@10 | ~75% | ~65% | ~88% |
| Latency (5 docs) | ~200ms | ~50ms | ~400ms |
| Citation accuracy | Low | Low | **High** |
| Best for | General queries | Keyword lookup | Production RAG |

---

## When to Choose Hybrid RAG

**Use Hybrid RAG when:**
- You need the highest answer quality and will accept slightly higher latency
- Your documents contain specific terminology (legal clauses, ticker symbols, regulation names)
- You need auditable answers with citations for compliance or review
- You're building a production-facing "ask my docs" system

**Consider other patterns when:**
- You need very low latency (< 100ms) → use Adaptive RAG
- Your question references entity relationships ("How does X affect Y?") → use Graph RAG
- Your documents contain charts or images → use Multimodal RAG
- You need full ReAct loop with tool use → use Agentic RAG

---

## How It Compares to CRAG

Both Hybrid RAG and CRAG improve retrieval quality, but in different ways:

| | CRAG | Hybrid RAG |
|---|---|---|
| **Focus** | Quality gating — *"are these docs good enough?"* | Better retrieval — *"find the right docs in the first place"* |
| **Method** | Score docs; retry or fall back to web | Combine BM25 + vector, rerank |
| **Citations** | Loose (just source list) | **Enforced** (inline citations required) |
| **Best for** | When document coverage is uncertain | When document coverage is good |
| **Combined** | ✓ Can be used together for maximum quality | ✓ |

---

## Implementation Notes

### BM25 index scope

The current implementation builds the BM25 index **over Qdrant search results** (top-50 candidates from vector search), not over the entire corpus. This means:
- ✓ Fast — no separate indexing step
- ✓ Scales well regardless of corpus size
- ~ Slight edge case: a document that scores zero on vector search but high on BM25 could be missed

For full corpus BM25, you would need to export all documents from Qdrant and build a persistent index. The current approach is practical for corpora up to ~50,000 chunks.

### The reranker is cached

The cross-encoder model is loaded once and cached as a singleton:

```python
@functools.lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    return CrossEncoder(model_name)
```

The first query will take ~2-3 seconds to load the model. All subsequent queries use the cached model.

---

## See Also

- [Citation Enforcement](../production/01_citation_enforcement.md) — How citations are structured and enforced
- [Prompt Versioning](../production/02_prompt_versioning.md) — How the citation prompt is versioned
- [CI/CD Evaluation Gating](../production/03_eval_ci_gating.md) — Testing Hybrid RAG quality in CI
- [Corrective RAG](04_corrective_rag.md) — Complementary quality-improvement pattern
