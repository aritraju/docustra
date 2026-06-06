# Self-RAG

## Overview

Self-RAG is designed for **high-stakes applications** where the cost of a hallucinated or unsupported answer is significant — legal, medical, compliance, or financial advisory contexts. It makes the LLM's reasoning process **transparent and auditable** by generating special reflection tokens at each step of the generation process.

Unlike all other RAG patterns where the LLM is a black box that takes context and produces an answer, Self-RAG produces an explicit decision trail that you can inspect, log, and audit.

---

## The Reflection Tokens

Self-RAG defines four special tokens that the LLM generates to critique itself:

| Token | Question Asked | Values |
|---|---|---|
| `[Retrieve]` | Does answering require external documents? | `YES` / `NO` |
| `[Relevant]` | Is this retrieved document relevant to the question? | `YES` / `NO` |
| `[Supported]` | Is the generated answer grounded in the context? | `YES` / `PARTIALLY` / `NO` |
| `[Useful]` | Is the final answer complete and useful? | `YES` / `NO` |

---

## Architecture

```
           User Question
                │
                ▼
    ┌───────────────────────────┐
    │  [Retrieve] Token          │
    │  "Does this need retrieval?"│
    └───────────┬───────────────┘
                │
       YES ─────┤───── NO
       │                │
       │            Direct answer
       │            (no retrieval)
       ▼
    ┌───────────────────────────┐
    │  VectorStore Search        │
    │  k=6 candidate documents   │
    └───────────┬───────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────┐
    │  [Relevant] Token per document                     │
    │                                                    │
    │  Doc 1: [Relevant] = YES  → keep                  │
    │  Doc 2: [Relevant] = NO   → discard               │
    │  Doc 3: [Relevant] = YES  → keep                  │
    │  Doc 4: [Relevant] = NO   → discard               │
    │  Doc 5: [Relevant] = YES  → keep                  │
    │  Doc 6: [Relevant] = NO   → discard               │
    └───────────┬───────────────────────────────────────┘
                │
                ▼ (only relevant docs)
    ┌───────────────────────────┐
    │  LLM Generate Answer       │
    │  from filtered context     │
    └───────────┬───────────────┘
                │
                ▼
    ┌───────────────────────────┐
    │  [Supported] Token         │
    │  "Is claim grounded?"      │
    │  YES / PARTIALLY / NO      │
    └───────────┬───────────────┘
                │
                ▼
    ┌───────────────────────────┐
    │  [Useful] Token            │
    │  "Is answer complete?"     │
    │  YES / NO                  │
    └───────────┬───────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────┐
    │  RAGResponse                                       │
    │  answer: "..."                                     │
    │  reasoning: "[Retrieve]: YES                       │
    │              [Relevant] Doc1: YES                  │
    │              [Relevant] Doc2: NO                   │
    │              [Supported]: PARTIALLY                │
    │              [Useful]: YES"                        │
    └───────────────────────────────────────────────────┘
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/self_rag.py`

### [Retrieve] Token

```python
_RETRIEVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Does answering this question require retrieving external documents? "
               "Answer YES or NO only."),
    ("human", "{question}"),
])

retrieve_raw = (_RETRIEVE_PROMPT | self._llm).invoke({"question": question}).content.strip().upper()
tokens.retrieve = "YES" in retrieve_raw
reasoning_log.append(f"[Retrieve]: {retrieve_raw}")
```

**Practical impact:** Questions like "What is the capital of France?" skip retrieval entirely. Questions like "What was Apple's R&D expense?" trigger retrieval. This saves token cost on knowledge-based questions.

### [Relevant] Token — Document Filtering

```python
_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Is this document relevant to answering the question? Answer YES or NO only."),
    ("human", "Question: {question}\n\nDocument: {document}"),
])

docs = self._vector_store.similarity_search(question, k=6)  # retrieve more, filter down
relevant_docs = []

for doc in docs:
    relevance = (_RELEVANCE_PROMPT | self._llm).invoke({
        "question": question,
        "document": doc.page_content[:500]
    }).content.strip().upper()
    
    if "YES" in relevance:
        relevant_docs.append(doc)
    reasoning_log.append(f"[Relevant] for chunk: {relevance}")
```

**Key design:** k=6 (retrieve more than needed) → filter to relevant subset. This is opposite to standard RAG which retrieves exactly k. Self-RAG trades token cost for answer quality.

### [Supported] Token — Grounding Verification

```python
_SUPPORTED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Is the answer fully supported by the provided context? "
               "Answer YES, PARTIALLY, or NO only."),
    ("human", "Context: {context}\n\nAnswer: {answer}"),
])

supported_raw = (_SUPPORTED_PROMPT | self._llm).invoke({
    "context": context[:1000],
    "answer": answer[:500]
}).content.strip().upper()

tokens.supported = "YES" in supported_raw or "PARTIALLY" in supported_raw
reasoning_log.append(f"[Supported]: {supported_raw}")
```

**Three-way classification:**
- `YES` — every claim in the answer is directly supported by retrieved text
- `PARTIALLY` — most claims supported, some inferred or extrapolated
- `NO` — significant claims not present in context → potential hallucination flag

### [Useful] Token

```python
_USEFUL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Is this a useful and complete answer to the question? Answer YES or NO only."),
    ("human", "Question: {question}\n\nAnswer: {answer}"),
])
```

If `[Useful] = NO`, a production system could automatically re-trigger retrieval with a different strategy.

---

## Demo Examples

**Document corpus:** Apple 10-K 2023

### Example 1 — Full Self-RAG Flow (All tokens fire)

**Request:**
```json
{
  "question": "What were Apple's total operating expenses in FY2023 and how do they break down?",
  "pattern": "self_rag"
}
```

**Self-RAG execution trace:**
```
[Retrieve]: YES — requires document lookup

Retrieving 6 candidate documents...

[Relevant] Doc 1 (revenue table, page 31): YES
[Relevant] Doc 2 (risk factors, page 8): NO
[Relevant] Doc 3 (operating expenses breakdown, page 34): YES  
[Relevant] Doc 4 (CEO letter, page 2): NO
[Relevant] Doc 5 (segment results, page 33): YES
[Relevant] Doc 6 (supply chain text, page 12): NO

3 relevant documents selected for generation

Generating answer from 3 filtered documents...

[Supported]: YES — all figures traceable to retrieved context
[Useful]: YES
```

**Response:**
```json
{
  "answer": "Apple's total operating expenses in FY2023 were $54,847 million, broken down as follows:\n\n- Research & Development: $26,251 million (6.8% of revenue)\n- Selling, General & Administrative: $24,932 million (6.5% of revenue)\n- Total operating expenses: $54,847 million (14.3% of revenue)\n\nNote: These figures represent operating expenses *excluding* cost of sales. Including cost of sales ($214,137M), total costs were $268,984M. Operating income was $114,301M (29.8% operating margin).\n\nR&D remained flat at $26.3B YoY (FY2022 was also $26.3B), while SG&A grew slightly from $25.1B, reflecting higher marketing and retail costs.",
  "pattern": "self_rag",
  "sources": [/* 3 relevant sources */],
  "reasoning": "[Retrieve]: YES\n[Relevant] for chunk: YES\n[Relevant] for chunk: NO\n[Relevant] for chunk: YES\n[Relevant] for chunk: NO\n[Relevant] for chunk: YES\n[Relevant] for chunk: NO\n[Supported]: YES\n[Useful]: YES",
  "metadata": {
    "tokens": {
      "retrieve": true,
      "relevant": true,
      "supported": true,
      "useful": true
    },
    "relevant_docs_count": 3
  }
}
```

---

### Example 2 — [Retrieve] = NO (Trivial Question)

**Request:**
```json
{
  "question": "What does R&D stand for?",
  "pattern": "self_rag"
}
```

**Self-RAG trace:**
```
[Retrieve]: NO — common knowledge, no retrieval needed
Direct generation (no vector search performed)
```

**Response:**
```json
{
  "answer": "R&D stands for Research and Development. In the context of corporate financial statements, R&D expense refers to costs incurred in developing new products, processes, or services, and improving existing ones.",
  "pattern": "self_rag",
  "reasoning": "[Retrieve]: NO — answered directly.",
  "metadata": {
    "tokens": {
      "retrieve": false,
      "relevant": true,
      "supported": true,
      "useful": true
    }
  }
}
```

---

### Example 3 — [Supported] = PARTIALLY (Audit Alert)

**Request:**
```json
{
  "question": "What is Apple's projected revenue for FY2024?",
  "pattern": "self_rag"
}
```

**Self-RAG trace:**
```
[Retrieve]: YES

Retrieved 6 documents — all mention FY2023 actuals, none contain FY2024 projections
3 deemed [Relevant] (FY2023 actuals and management outlook language)

Generated answer includes: "Analysts project ~$400B for FY2024..." ← inferred, not in docs

[Supported]: PARTIALLY ← correctly identified that projection is not in corpus
[Useful]: YES
```

**Response:**
```json
{
  "answer": "The 10-K filing (as a historical document) does not provide revenue projections for FY2024. What it does disclose is management's qualitative outlook: Apple expects continued Services growth and sees iPhone demand as stable in developed markets. Based on FY2023 actual revenue of $383.3B and management commentary, analysts typically model modest revenue recovery. However, these projections are not from the filing itself.",
  "pattern": "self_rag",
  "reasoning": "[Retrieve]: YES\n[Relevant] ...\n[Supported]: PARTIALLY — projection not in retrieved context, extrapolated\n[Useful]: YES",
  "metadata": {
    "tokens": {
      "retrieve": true,
      "relevant": true,
      "supported": false,
      "useful": true
    }
  }
}
```

**This is the key value of Self-RAG:** The `[Supported]: PARTIALLY` flag in the reasoning field tells the caller that part of the answer involves inference beyond the document corpus. A compliance system could flag this response for human review.

---

### Example 4 — All Docs Deemed Irrelevant

**Request:**
```json
{
  "question": "What is Apple's carbon footprint?",
  "pattern": "self_rag"
}
```

**Self-RAG trace:**
```
[Retrieve]: YES

Retrieved 6 documents:
[Relevant]: NO (revenue data)
[Relevant]: NO (supply chain text)
[Relevant]: NO (legal proceedings)
[Relevant]: NO (employee headcount)
[Relevant]: NO (segment results)
[Relevant]: NO (risk factors - general)

All 6 deemed irrelevant → fallback to original 3 docs
[Supported]: NO ← answer cannot be supported
[Useful]: YES (partial answer given)
```

**Response:**
```json
{
  "answer": "The indexed sections of the Apple 10-K filing do not contain specific carbon footprint data. Apple's environmental commitments (including a goal of carbon neutrality across its supply chain by 2030) are disclosed in their separate Environmental Progress Report and ESG disclosures, which are not part of the annual 10-K filing indexed here. To find this data, please ingest Apple's Environmental Progress Report.",
  "reasoning": "[Relevant]: NO for all 6 docs — no relevant documents found, using all retrieved.\n[Supported]: NO\n[Useful]: YES",
  "metadata": {
    "tokens": {"retrieve": true, "relevant": false, "supported": false, "useful": true}
  }
}
```

---

## The Audit Trail

The `reasoning` field in every Self-RAG response is a machine-readable audit trail. A compliance dashboard could parse this and:

```python
import json

response = query_api(question, pattern="self_rag")
tokens = response["metadata"]["tokens"]

if not tokens["supported"]:
    flag_for_human_review(response, reason="Unsupported claims in answer")

if not tokens["relevant"]:
    log_retrieval_gap(question, reason="No relevant documents found")

if tokens["retrieve"] and not tokens["useful"]:
    escalate_to_agentic_rag(question, reason="Useful token failed")
```

---

## Configuration

```env
RETRIEVAL_TOP_K=6    # retrieve more than needed — Self-RAG filters down
LLM_PROVIDER=gemini  # each token check is a separate LLM call
```

**Token check cost:** Self-RAG makes approximately 2 + k additional LLM calls compared to standard RAG (1 [Retrieve], k [Relevant] per doc, 1 [Supported], 1 [Useful]). With k=6, that's 10 additional calls.

**Cost optimization:** Reduce [Relevant] checks to the first 3 documents only:
```python
for doc in docs[:3]:  # score only top-3 instead of all k
```

---

## When to Use Self-RAG

**Use when:**
- Answers will be acted upon (financial decisions, legal filings, medical guidance)
- Auditability and explainability are regulatory requirements
- You need to detect and flag hallucinations automatically
- Building systems where human review is triggered conditionally

**Avoid when:**
- Latency is critical (8-12 LLM calls per query)
- Casual Q&A where the cost of a wrong answer is low
- High query volume where the token cost multiplier is prohibitive

---

## Token Cost Comparison

| Pattern | LLM Calls per Query | Relative Token Cost |
|---|---|---|
| Standard RAG | 1 | 1× |
| HyDE | 2 | 2× |
| Adaptive (simple) | 2 | 2× |
| CRAG (pass) | 2 | 2× |
| Branched (3 branches) | 5 | 5× |
| **Self-RAG** | **8-12** | **8-12×** |
| Agentic (3 iterations) | 6-10 | 6-10× |

Self-RAG is the most expensive per-query pattern. Its value is in **quality insurance**, not efficiency.

---

## Related Patterns

- **CRAG** — cheaper alternative for quality guarantees; less granular
- **Agentic RAG** — if [Useful] = NO, automatically escalate to Agentic for another attempt
- **Adaptive RAG** — route high-stakes queries to Self-RAG, routine queries to Adaptive
