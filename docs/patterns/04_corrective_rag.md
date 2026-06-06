# Corrective RAG (CRAG)

## Overview

Corrective RAG addresses one of the most common failure modes in production RAG systems: **retrieved documents that are irrelevant to the query**. When a standard RAG pipeline retrieves low-quality context, the LLM either hallucinates to fill the gap or produces a vague, non-committal answer.

CRAG adds an **evaluation step after retrieval**. If retrieved documents score below a confidence threshold, the system automatically corrects itself by either reformulating the query or falling back to a live web search — before generating the final answer.

---

## The Problem It Solves

```
Standard RAG failure:
  Query: "What did the FASB ASC 606 standard change in Apple's revenue recognition?"
  Retrieved: [Generic revenue docs from Apple 10-K — no mention of ASC 606]
  LLM: "Apple follows standard revenue recognition practices..." ← hallucination
  
CRAG behavior:
  Same query → relevance score: 0.31 (below threshold 0.7)
  → Query rewritten: "FASB ASC 606 revenue from contracts with customers Apple adoption impact"
  → Retry or web search → finds specific ASC 606 disclosure
  → Correct, grounded answer
```

---

## Architecture

```
          User Question
                │
                ▼
    ┌───────────────────────┐
    │  VectorStore Search    │
    │  with relevance scores │
    └───────────┬───────────┘
                │
    ┌───────────▼──────────────────────────────────────────┐
    │            Relevance Evaluation                       │
    │                                                       │
    │   avg_score = mean(scores for all retrieved docs)     │
    │                                                       │
    │   avg_score ≥ threshold (0.7)?                        │
    └────────────────┬─────────────────┬────────────────────┘
                     │ YES             │ NO
                     │                 ▼
                     │    ┌────────────────────────┐
                     │    │  LLM Query Rewriter     │
                     │    │  (reformulate query)    │
                     │    └─────────────┬──────────┘
                     │                  │
                     │    ┌─────────────▼──────────┐
                     │    │  Retry Vector Search    │
                     │    └─────────────┬──────────┘
                     │                  │
                     │    retry_score ≥ threshold?
                     │        YES │          │ NO
                     │            │          ▼
                     │            │  ┌───────────────┐
                     │            │  │ Tavily Web    │
                     │            │  │ Search        │
                     │            │  └───────┬───────┘
                     │            │          │
    Use original     Use retry    │  Use web results
    docs             docs         │
                     └────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │  LLM Generation  │
                    └─────────────────┘
                           │
                           ▼
                      RAGResponse
                  + fallback metadata
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/corrective.py`

### Step 1 — Retrieval with Scores

```python
def query(self, question: str) -> RAGResponse:
    # Returns (Document, relevance_score) tuples
    docs_with_scores = self._vector_store.similarity_search_with_scores(question)
    
    avg_score = (
        sum(s for _, s in docs_with_scores) / len(docs_with_scores)
        if docs_with_scores else 0.0
    )
```

Qdrant's cosine similarity scores are normalized to [0, 1]. A score of 0.7 means the document shares ~70% semantic similarity with the query vector.

### Step 2 — Threshold Decision

```python
if avg_score >= self._threshold:
    docs = [d for d, _ in docs_with_scores]
    fallback_used = "vector_search"
else:
    # Trigger correction flow
    rewritten = self._rewrite_query(question)
    retry_docs = self._vector_store.similarity_search(rewritten, k=5)
    retry_score = self._score_docs(rewritten, retry_docs)
    
    if retry_score >= self._threshold or not self._web_search:
        docs = retry_docs
        fallback_used = "rewritten_vector_search"
    else:
        docs = self._web_search_fallback(question)
        fallback_used = "web_search"
```

### Step 3 — LLM-based Re-scoring

After rewriting, individual document relevance is scored by the LLM for finer granularity:

```python
_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Score the relevance of this document to the question.
Return ONLY a number between 0.0 and 1.0. Nothing else."""),
    ("human", "Question: {question}\n\nDocument: {document}"),
])

def _score_docs(self, question: str, docs: list[Document]) -> float:
    scores = []
    for doc in docs[:3]:  # score first 3 to control cost
        raw = (_RELEVANCE_PROMPT | self._llm).invoke({
            "question": question,
            "document": doc.page_content[:500]
        }).content
        scores.append(float(raw.strip()))
    return sum(scores) / len(scores)
```

**Why LLM scoring after rewrite?** Qdrant's vector similarity score uses embedding distance, which can miss semantic nuance. The LLM relevance score is slower but more accurate for borderline cases.

### Step 4 — Query Rewriting

```python
_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question to improve document retrieval. "
               "Be more specific and use different keywords."),
    ("human", "{question}"),
])

def _rewrite_query(self, question: str) -> str:
    return (_REWRITE_PROMPT | self._llm).invoke({"question": question}).content.strip()
```

**Rewrite examples:**

| Original | Rewritten |
|---|---|
| "What accounting rules apply?" | "GAAP FASB accounting standards revenue recognition Apple financial statements" |
| "Any China issues?" | "Apple China supply chain manufacturing risk Foxconn geopolitical disruption" |
| "Employee info" | "Apple headcount full-time employees workforce FY2023 annual report" |

---

## Demo Examples

**Document corpus:** Apple 10-K 2023

### Example 1 — High Relevance (passes threshold)

**Request:**
```json
{
  "question": "What was Apple's total revenue from iPhone in FY2023?",
  "pattern": "corrective"
}
```

**CRAG evaluation:**
```
Retrieved docs avg relevance score: 0.87 (threshold: 0.70)
Decision: ✅ Use retrieved docs directly
Fallback: vector_search
```

**Response:**
```json
{
  "answer": "Apple's iPhone net revenue in FY2023 was $200,583 million ($200.6 billion), representing 52.3% of total net revenues of $383,285 million. This was a slight decrease from $205,489 million in FY2022.",
  "pattern": "corrective",
  "reasoning": "Avg relevance score: 0.87 (threshold: 0.70). Fallback used: vector_search.",
  "metadata": {
    "avg_score": 0.87,
    "fallback": "vector_search"
  }
}
```

---

### Example 2 — Low Relevance → Query Rewrite

**Request:**
```json
{
  "question": "How does Apple handle lease accounting?",
  "pattern": "corrective"
}
```

**CRAG evaluation:**
```
Initial retrieval avg relevance score: 0.41 (threshold: 0.70)
Decision: ⚠️ Score too low — rewriting query
Rewritten query: "Apple operating finance lease right-of-use assets liabilities ASC 842 accounting policy"
Retry avg relevance score: 0.78 ✅
Fallback: rewritten_vector_search
```

**Response:**
```json
{
  "answer": "Apple accounts for leases under ASC 842, recording right-of-use (ROU) assets and corresponding lease liabilities on the balance sheet. As of September 2023, Apple had: Operating lease ROU assets of $10,661M and finance lease ROU assets of $1,069M. Operating lease liabilities totaled $12,842M ($1,738M current, $11,104M non-current). Leases primarily cover retail stores, data centers, and corporate facilities with remaining terms of 1-20 years.",
  "pattern": "corrective",
  "reasoning": "Avg relevance score: 0.41 (threshold: 0.70). Fallback used: rewritten_vector_search.",
  "metadata": {
    "avg_score": 0.41,
    "fallback": "rewritten_vector_search"
  }
}
```

---

### Example 3 — Low Relevance → Web Search Fallback

When the document corpus doesn't contain the answer at all:

**Request:**
```json
{
  "question": "What is Apple's latest ESG sustainability rating from MSCI?",
  "pattern": "corrective"
}
```

**CRAG evaluation:**
```
Initial retrieval avg relevance score: 0.23 (threshold: 0.70)
Rewritten: "Apple MSCI ESG rating sustainability score AAA AA"
Retry relevance score: 0.19 — still below threshold
Decision: 🌐 Falling back to Tavily web search
Fallback: web_search
```

**Response:**
```json
{
  "answer": "Apple holds an MSCI ESG Rating of 'AA' (as of 2024), placing it in the 'Leader' category among technology companies. This reflects Apple's strong environmental commitments (carbon neutrality across supply chain by 2030), governance practices, and social policies. Note: This information comes from a web search as the 10-K filing does not cite third-party ESG ratings.",
  "pattern": "corrective",
  "reasoning": "Avg relevance score: 0.23 (threshold: 0.70). Fallback used: web_search.",
  "metadata": {
    "avg_score": 0.23,
    "fallback": "web_search"
  }
}
```

---

### Example 4 — Document Coverage Gap (graceful degradation)

When `TAVILY_API_KEY` is not set:

**CRAG behavior without web search:**
```
Score: 0.21 → below threshold
Rewrite → retry score: 0.25 → still below threshold
No Tavily configured → use retry docs anyway (best available)
Fallback: rewritten_vector_search
Answer: "Based on available documents, I could not find specific information about..." 
```

CRAG always returns *something* — it degrades gracefully rather than throwing an error.

---

## Configuration

```env
RETRIEVAL_SCORE_THRESHOLD=0.7    # main threshold — tune based on your corpus
RETRIEVAL_TOP_K=5                # more docs = more accurate avg score
TAVILY_API_KEY=your_key          # optional: enables web fallback
```

**Tuning the threshold:**

| Threshold | Effect |
|---|---|
| 0.5 | Rarely triggers correction; more hallucination risk |
| 0.7 | Balanced (recommended for most corpora) |
| 0.85 | Very strict; triggers rewrite/web for most queries |
| 0.95 | Over-triggers; web search becomes primary |

Run a calibration pass: sample 50 queries, record scores, inspect answers at each threshold.

---

## When to Use CRAG

**Use when:**
- Document coverage is uneven (some topics well-indexed, others sparse)
- Answer quality is more important than latency
- You have a Tavily API key for web fallback on out-of-corpus queries
- The cost of a wrong answer is high (finance, legal, compliance)

**Avoid when:**
- All queries are expected to be well-covered by the corpus (saves LLM scoring cost)
- Latency is critical (correction path adds 1-2 LLM calls)
- Web search fallback is not acceptable (sensitive/air-gapped environments)

---

## Scoring Decision Tree

```
                      ┌─────────────────┐
                      │ avg_score ≥ 0.7? │
                      └────────┬────────┘
                     YES ──────┤──────── NO
                     │                   │
               Use docs             Rewrite query
               directly             ┌───▼────────────┐
                                    │ retry ≥ 0.7?    │
                                    └────┬────────────┘
                                YES ─────┤──────── NO
                                │                   │
                          Use retry docs     Tavily configured?
                                            YES ──── NO
                                             │       │
                                          Web      Use retry
                                          search   docs anyway
```

---

## Related Patterns

- **Self-RAG** — evaluates relevance per-document with explicit tokens; slower but more granular
- **Adaptive RAG** — CRAG can be the quality gate inside the complex retrieval path
- **Agentic RAG** — the agent inherently performs correction by searching again when dissatisfied
