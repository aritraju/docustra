# Adaptive RAG

## Overview

Adaptive RAG solves a fundamental inefficiency in standard RAG pipelines: **treating every query the same way regardless of complexity**. A question like "What is 2+2?" doesn't need document retrieval. A question like "Compare Apple's debt-to-equity ratio trends with the industry average cited in the filing" needs deep multi-step retrieval.

Adaptive RAG inserts a **routing layer** before retrieval that classifies query complexity and dispatches to the appropriate pipeline depth.

---

## The Problem It Solves

| Without Adaptive RAG | With Adaptive RAG |
|---|---|
| Every query hits the vector store | Trivial queries bypass retrieval entirely |
| Same latency for simple and complex queries | Latency proportional to complexity |
| Wasted LLM tokens on unnecessary context | Context retrieved only when needed |
| No cost optimization | ~40-60% cost reduction on typical query mixes |

---

## Architecture

```
                    ┌─────────────────────┐
                    │   User Question      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Classifier     │
                    │  (1 LLM call,        │
                    │   ~50 tokens)        │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          "trivial"         "simple"        "complex"
               │               │               │
               ▼               ▼               ▼
        ┌──────────┐   ┌──────────────┐  ┌──────────────────┐
        │ Direct   │   │ 1× Vector    │  │ LLM Decomposer   │
        │ LLM      │   │ Search       │  │ → N sub-questions│
        │ Answer   │   │ → Answer     │  │ → N× searches    │
        └──────────┘   └──────────────┘  │ → Synthesize     │
                                         └──────────────────┘
               │               │               │
               └───────────────┴───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     RAGResponse      │
                    │  + complexity label  │
                    └─────────────────────┘
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/adaptive.py`

### Step 1 — Query Classification

The classifier is a single LLM call with a constrained output format:

```python
_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user question into one of three categories:
- trivial: common knowledge, math, definitions that need no document lookup
- simple: factual lookup from documents, single concept
- complex: multi-hop reasoning, comparisons, or questions spanning multiple concepts

Respond with exactly one word: trivial, simple, or complex."""),
    ("human", "{question}"),
])

def _classify(self, question: str) -> QueryComplexity:
    chain = _ROUTER_PROMPT | self._llm
    result = chain.invoke({"question": question}).content.strip().lower()
    return QueryComplexity(result)  # Enum: TRIVIAL | SIMPLE | COMPLEX
```

**Design choice:** Constraining to one word prevents the LLM from generating a paragraph of reasoning, keeping the routing call fast and cheap.

### Step 2 — Trivial Path

No retrieval whatsoever — direct generation:

```python
def _answer_directly(self, question: str) -> RAGResponse:
    chain = _DIRECT_PROMPT | self._llm
    answer = chain.invoke({"question": question}).content
    return RAGResponse(
        answer=answer,
        reasoning="Classified as trivial — answered without retrieval.",
        metadata={"complexity": "trivial"},
    )
```

### Step 3 — Simple Path

Single vector search, standard RAG:

```python
def _simple_retrieval(self, question: str) -> RAGResponse:
    docs = self._vector_store.similarity_search(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (_RAG_PROMPT | self._llm).invoke(
        {"context": context, "question": question}
    ).content
    return RAGResponse(answer=answer, sources=self._format_sources(docs), ...)
```

### Step 4 — Complex Path

Decompose → parallel sub-retrievals → synthesize:

```python
def _complex_retrieval(self, question: str) -> RAGResponse:
    # Decompose into 2-3 sub-questions
    sub_questions = self._decompose(question)
    
    # Retrieve for each sub-question
    all_docs = []
    for sq in sub_questions:
        all_docs.extend(self._vector_store.similarity_search(sq, k=3))
    
    # Deduplicate
    unique_docs = deduplicate(all_docs)
    
    # Single synthesis call
    answer = synthesize(unique_docs, question)
    return RAGResponse(answer=answer, metadata={"sub_questions": sub_questions})
```

---

## Demo Example

**Document corpus:** Apple 10-K 2023 filing

### Trivial Query

**Request:**
```json
{
  "question": "What does SEC stand for?",
  "pattern": "adaptive"
}
```

**Response:**
```json
{
  "answer": "SEC stands for Securities and Exchange Commission, the U.S. federal agency responsible for regulating securities markets and protecting investors.",
  "pattern": "adaptive",
  "sources": [],
  "reasoning": "Classified as trivial — answered without retrieval.",
  "metadata": {
    "complexity": "trivial"
  }
}
```

**LLM calls:** 2 (1 classify + 1 answer). No vector search.

---

### Simple Query

**Request:**
```json
{
  "question": "What was Apple's total net revenue in fiscal year 2023?",
  "pattern": "adaptive"
}
```

**Response:**
```json
{
  "answer": "Apple's total net revenue in fiscal year 2023 was $383.3 billion, a decrease of 2.8% compared to $394.3 billion in fiscal year 2022.",
  "pattern": "adaptive",
  "sources": [
    {
      "content": "Total net revenue: $383,285 million...",
      "source": "aapl-20230930.pdf",
      "page": 31
    }
  ],
  "reasoning": "Classified as simple — single-step vector retrieval.",
  "metadata": {
    "complexity": "simple"
  }
}
```

**LLM calls:** 2 (1 classify + 1 answer). 1 vector search.

---

### Complex Query

**Request:**
```json
{
  "question": "How do Apple's supply chain risks in China relate to their revenue concentration and what hedging strategies do they employ?",
  "pattern": "adaptive"
}
```

**Response:**
```json
{
  "answer": "Apple's exposure to China-based manufacturing creates compounded risk across three dimensions. First, approximately 95% of iPhone assembly occurs in China through Foxconn and Pegatron, creating concentration risk. Second, China represents Apple's third-largest market at $72.6B (19% of revenue) in FY2023, meaning disruptions affect both supply and demand simultaneously. To mitigate this, Apple employs: (1) supplier diversification to India and Vietnam, (2) natural hedging through local currency revenue matching local costs, and (3) forward contracts to hedge foreign exchange exposure on non-functional currency cash flows.",
  "pattern": "adaptive",
  "sources": [
    {"content": "Our operations and performance depend significantly...", "source": "aapl-20230930.pdf", "page": 8},
    {"content": "Greater China segment net revenues...", "source": "aapl-20230930.pdf", "page": 33},
    {"content": "We may enter into foreign currency forward and option contracts...", "source": "aapl-20230930.pdf", "page": 52}
  ],
  "reasoning": "Classified as complex — decomposed into 3 sub-questions.",
  "metadata": {
    "complexity": "complex",
    "sub_questions": [
      "What are Apple's key supply chain risks related to China?",
      "What is Apple's revenue concentration in China?",
      "What hedging strategies does Apple use for currency and supply chain risk?"
    ]
  }
}
```

**LLM calls:** 5 (1 classify + 1 decompose + 3 sub-retrievals + 1 synthesize). 3 vector searches in sequence.

---

## Configuration

```env
# .env
RETRIEVAL_TOP_K=5          # docs per vector search
CHUNK_SIZE=512             # affects retrieval granularity
LLM_PROVIDER=gemini        # classifier and answerer
```

No additional configuration needed — the classifier uses the same LLM as the answerer.

---

## When to Use Adaptive RAG

**Use when:**
- Your query mix is diverse (some simple, some complex)
- Cost efficiency matters — you're on a free API tier with rate limits
- You want a general-purpose RAG endpoint that "just works"

**Avoid when:**
- All queries are known to be complex (skip the routing overhead)
- Latency is ultra-critical (the classification call adds ~200-400ms)
- Query type is deterministic (use the specific pattern directly)

---

## Performance Characteristics

| Complexity | LLM Calls | Vector Searches | Typical Latency |
|---|---|---|---|
| Trivial | 2 | 0 | ~500ms |
| Simple | 2 | 1 | ~800ms |
| Complex | 4-6 | 2-4 | ~2-4s |

---

## Evaluation Tips

When benchmarking Adaptive RAG, log the `complexity` field per query and compute:
- **Routing accuracy:** Does the classifier route correctly? (Sample 20 queries, label manually)
- **Trivial bypass rate:** What % of queries skip retrieval? (Expect 15-30% on enterprise corpora)
- **Faithfulness per tier:** RAGAS faithfulness should be high for simple/complex, less relevant for trivial

---

## Related Patterns

- **Branched RAG** — the complex path in Adaptive RAG is a simplified version of Branched RAG
- **Corrective RAG** — can be composed with Adaptive: route complex queries to CRAG for quality guarantees
- **Agentic RAG** — the most powerful complex-path alternative; higher latency, higher quality
