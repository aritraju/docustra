# Branched RAG

## Overview

Branched RAG decomposes a complex question into **multiple independent sub-questions**, retrieves context for each in **parallel**, and then synthesizes the results into a single coherent answer. Unlike Agentic RAG's dynamic loop, Branched RAG follows a **predetermined structure**: decompose → fan-out → fan-in.

This pattern excels at comparative, multi-faceted, or analytical questions that require information from several distinct sections of a document corpus simultaneously.

---

## The Problem It Solves

Standard RAG uses one query vector to retrieve documents. For a question like:

> "Compare Apple's operating expenses, headcount changes, and capital expenditure trends from FY2021 to FY2023"

A single embedding captures some aspects of this question but misses others. The retrieved documents will likely cover one or two dimensions but not all three, producing an incomplete answer.

Branched RAG fires a separate, focused retrieval for each dimension, ensuring full coverage.

---

## Architecture

```
           User Question
                │
                ▼
    ┌───────────────────────┐
    │    LLM Decomposer      │
    │  → sub_q1              │
    │  → sub_q2              │
    │  → sub_q3              │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────────────────────────┐
    │         ThreadPoolExecutor                 │
    │  (parallel branch execution)               │
    │                                            │
    │  Branch 1          Branch 2   Branch 3     │
    │  ┌──────────┐   ┌──────────┐ ┌──────────┐ │
    │  │vector    │   │vector    │ │vector    │ │
    │  │search    │   │search    │ │search    │ │
    │  │(sub_q1)  │   │(sub_q2)  │ │(sub_q3)  │ │
    │  └────┬─────┘   └────┬─────┘ └────┬─────┘ │
    │       │LLM answer    │LLM answer  │LLM ans │
    └───────┼──────────────┼────────────┼────────┘
            │              │            │
            └──────────────┴────────────┘
                           │
                           ▼
                ┌────────────────────┐
                │   LLM Synthesizer   │
                │   (fan-in)          │
                └────────────────────┘
                           │
                           ▼
                      RAGResponse
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/branched.py`

### Step 1 — Decomposition

```python
_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Decompose the question into 2-4 independent sub-questions that together
cover the full answer. Each sub-question should be answerable independently.
Return ONLY a numbered list, one sub-question per line."""),
    ("human", "{question}"),
])

def _decompose(self, question: str) -> list[str]:
    chain = _DECOMPOSE_PROMPT | self._llm
    raw = chain.invoke({"question": question}).content.strip()
    lines = [l.split(". ", 1)[-1].strip() for l in raw.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 10][:4]  # max 4 branches
```

For the question *"Compare Apple's operating expenses, headcount, and capex trends FY2021-2023"*, the decomposer produces:

```
1. What were Apple's operating expenses in FY2021, FY2022, and FY2023?
2. How did Apple's headcount change from FY2021 to FY2023?
3. What were Apple's capital expenditure figures from FY2021 to FY2023?
```

### Step 2 — Parallel Branch Execution

```python
def _answer_branch(self, sub_question: str) -> tuple[str, list]:
    docs = self._vector_store.similarity_search(sub_question, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (_BRANCH_PROMPT | self._llm).invoke(
        {"context": context, "question": sub_question}
    ).content
    return answer, docs

# Run all branches in parallel
with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
    futures = {executor.submit(self._answer_branch, sq): sq for sq in sub_questions}
    branch_results = []
    for future, sq in futures.items():
        answer, docs = future.result()
        branch_results.append({"sub_question": sq, "answer": answer})
```

**Why `ThreadPoolExecutor` and not `asyncio`?**
LangChain's `invoke()` is synchronous. `ThreadPoolExecutor` achieves true parallelism for I/O-bound LLM calls without requiring async refactoring. With 3 branches on Gemini, this reduces wall-clock time by ~60% vs. sequential execution.

### Step 3 — Synthesis

```python
_SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You have answers to several sub-questions of a complex query.
Synthesize them into a single coherent, comprehensive answer.
Do not repeat information. Cite sources where relevant."""),
    ("human", "Original question: {original_question}\n\nSub-question answers:\n{sub_answers}"),
])

sub_answers_text = "\n\n".join(
    f"Q: {r['sub_question']}\nA: {r['answer']}" for r in branch_results
)
final_answer = (_SYNTHESIZE_PROMPT | self._llm).invoke({
    "original_question": question,
    "sub_answers": sub_answers_text
}).content
```

---

## Demo Examples

**Document corpus:** Apple 10-K 2023 filing

### Example 1 — Financial Comparison Query

**Request:**
```json
{
  "question": "Compare Apple's gross margin, operating margin, and net margin for FY2023 and explain the trends.",
  "pattern": "branched"
}
```

**Decomposition:**
```
Branch 1: What was Apple's gross profit and gross margin percentage in FY2023?
Branch 2: What was Apple's operating income and operating margin in FY2023?
Branch 3: What was Apple's net income and net profit margin in FY2023?
```

**Branch answers (parallel execution, ~2.1s):**
```
Branch 1: Gross profit: $169,148M on revenue of $383,285M → gross margin: 44.1%
          (up from 43.3% in FY2022, driven by services mix)

Branch 2: Operating income: $114,301M → operating margin: 29.8%
          (down from 30.3% in FY2022 due to higher R&D and SG&A)

Branch 3: Net income: $96,995M → net margin: 25.3%
          (down from 25.3% in FY2022, essentially flat)
```

**Final synthesized response:**
```json
{
  "answer": "Apple's FY2023 profitability metrics show a nuanced picture:\n\n**Gross Margin: 44.1%** (up from 43.3% in FY2022)\nThe improvement reflects Apple's ongoing mix shift toward high-margin Services revenue (App Store, iCloud, Apple Music), which carries ~70%+ gross margins compared to ~35% for hardware.\n\n**Operating Margin: 29.8%** (down from 30.3% in FY2022)\nDespite gross margin improvement, operating margin compressed slightly due to R&D spending remaining flat at $26.3B (+0% YoY) while revenue declined 2.8%. This increased R&D as a % of revenue.\n\n**Net Margin: 25.3%** (flat vs. FY2022)\nNet margin held steady as lower operating income was offset by higher other income/interest from Apple's substantial cash position.\n\n**Key trend:** Apple is successfully executing a margin-expansion strategy through Services, but hardware revenue decline and fixed cost base create near-term margin pressure at the operating level.",
  "pattern": "branched",
  "sources": [/* 9 sources from 3 branches */],
  "reasoning": "Decomposed into 3 branches, retrieved in parallel, then synthesized.",
  "metadata": {
    "sub_questions": [
      "What was Apple's gross profit and gross margin percentage in FY2023?",
      "What was Apple's operating income and operating margin in FY2023?",
      "What was Apple's net income and net profit margin in FY2023?"
    ],
    "branch_answers": [/* per-branch answers */]
  }
}
```

---

### Example 2 — Risk Factor Multi-Dimension Query

**Request:**
```json
{
  "question": "What are Apple's key risk factors related to competition, regulation, and supply chain as disclosed in the 10-K?",
  "pattern": "branched"
}
```

**Decomposition:**
```
Branch 1: What competitive risks does Apple disclose in its 10-K?
Branch 2: What regulatory and legal risks does Apple face according to the 10-K?
Branch 3: What supply chain and manufacturing risks does Apple disclose?
```

**Response (abbreviated):**
```json
{
  "answer": "Apple's 10-K discloses material risks across three dimensions:\n\n**Competition Risks:**\n- Intense competition in all markets with competitors having greater resources in certain segments\n- Rapid technological change requiring continuous innovation\n- Price competition from lower-cost Android manufacturers affecting market share\n\n**Regulatory Risks:**\n- EU Digital Markets Act designating Apple as a 'gatekeeper' — requiring third-party app stores and alternate payment systems in the EU\n- Ongoing DOJ and state AG antitrust investigations into App Store practices\n- Data privacy regulations (GDPR, CCPA) increasing compliance costs\n- Potential forced changes to App Store economics affecting ~$85B Services revenue\n\n**Supply Chain Risks:**\n- ~95% of iPhone production concentrated in China (primarily Foxconn/Hon Hai)\n- Geopolitical tensions between US and China creating disruption risk\n- Single-source components for certain technologies\n- Long lead times for specialized semiconductors (TSMC dependency for Apple Silicon)\n\nThe regulatory risks represent the most material near-term threat to Apple's business model given active enforcement actions.",
  "pattern": "branched",
  "reasoning": "Decomposed into 3 branches, retrieved in parallel, then synthesized.",
  "metadata": {
    "sub_questions": [
      "What competitive risks does Apple disclose?",
      "What regulatory and legal risks does Apple face?",
      "What supply chain and manufacturing risks does Apple disclose?"
    ]
  }
}
```

---

### Example 3 — Temporal Comparison Query

**Request:**
```json
{
  "question": "How did Apple's cash position, long-term debt, and share buyback program change from FY2021 to FY2023?",
  "pattern": "branched"
}
```

**Branch execution timeline (parallel):**

```
t=0ms   ─── Branch 1 starts: "Apple cash and cash equivalents 2021 2022 2023"
t=0ms   ─── Branch 2 starts: "Apple long-term debt 2021 2022 2023"
t=0ms   ─── Branch 3 starts: "Apple share repurchase buyback program 2021 2022 2023"
t=1,840ms ─ Branch 2 returns (fastest)
t=2,100ms ─ Branch 1 returns
t=2,310ms ─ Branch 3 returns  ← all branches done, begin synthesis
t=3,800ms ─ Synthesis complete → response returned
```

vs. sequential execution: ~6,200ms

---

## Timing Benchmark

```
Query: "Compare Apple's margins, risks, and capital allocation"
Branches: 3

Sequential execution:   6.2s
Parallel execution:     2.4s
Speedup:               2.6×
```

The speedup increases with more branches and slower LLM providers.

---

## Configuration

```env
RETRIEVAL_TOP_K=3    # docs per branch (lower than default to control context size)
LLM_PROVIDER=gemini  # Gemini Flash handles parallel calls well within free tier
```

**Controlling max branches** (edit `branched.py`):
```python
return [l for l in lines if len(l) > 10][:4]  # change 4 to your limit
```

---

## When to Use Branched RAG

**Use when:**
- The question explicitly contains multiple distinct dimensions ("compare X, Y, and Z")
- Questions about temporal trends (FY2021 vs FY2022 vs FY2023)
- Risk/opportunity analysis that spans multiple categories
- Any "multi-part question" structure is visible in the query

**Avoid when:**
- Sub-questions are highly interdependent (answer to Q2 depends on Q1's result → use Agentic)
- The question is simple and factual (unnecessary overhead)
- The LLM decomposition quality is poor for your domain (validate with test queries)

---

## Common Pitfall: Dependent Sub-questions

Branched RAG assumes sub-questions are **independent**. This fails when:

```
Q: "What were Apple's R&D expenses, and what % of revenue did that represent?"
```

Sub-question 2 depends on the answer to sub-question 1. In this case:
- Use **Adaptive RAG** (it handles sequential multi-hop)
- Or use **Agentic RAG** (agent discovers dependencies dynamically)

---

## Related Patterns

- **Adaptive RAG** — uses a simplified version of Branched for its "complex" path
- **Agentic RAG** — handles dependent sub-questions; higher latency
- **HyDE** — can be combined: generate a hypothetical doc per branch for better retrieval
