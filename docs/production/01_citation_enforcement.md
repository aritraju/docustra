# Citation Enforcement

**Available in:** All RAG patterns (via shared prompt) | **Strictest in:** Hybrid RAG

---

## Overview

Citation enforcement means the system **must prove every answer** by pointing to the exact passage it came from. If it can't prove an answer, it must say so instead of guessing.

This solves the most dangerous failure mode in RAG systems: **silent hallucination** — where the model sounds confident but is making things up.

---

## The Problem: Silent Hallucination

Without citation enforcement, a standard RAG system behaves like this:

```
Question: "What was Apple's net income in fiscal 2023?"

Retrieved context (somewhat relevant but not specific):
  "Apple reported strong financial results for fiscal year 2023,
   with growth across all segments..."

LLM answer (hallucinated):
  "Apple's net income in fiscal 2023 was $94 billion."
  
  ← This number is fabricated. The context doesn't contain it.
     But the answer sounds confident and specific.
```

A user reading this has no way to know it's wrong unless they go and check the original document. In a legal, financial, or medical context, this is dangerous.

---

## The Solution: Citation-Enforced Answers

With citation enforcement:

```
Question: "What was Apple's net income in fiscal 2023?"

Retrieved context (contains the answer):
  [Passage 1 | Source: apple_10k_2023.pdf, Page: 52]
  "The Company's net income for fiscal 2023 was $97.0 billion,
   compared to $99.8 billion in fiscal 2022."

LLM answer (grounded):
  "Apple's net income for fiscal 2023 was $97.0 billion
   [Source: apple_10k_2023.pdf, Page: 52]."
  
  ← Every number is traceable to a specific page.
```

If the answer is NOT in the retrieved context:

```
Question: "What is Apple's secret formula for A17 Pro chips?"

Retrieved context (no relevant passages)

LLM answer (decline):
  "I cannot answer this question based on the provided documents.
   The retrieved context does not contain sufficient information."
  
  ← Honest refusal. No hallucination.
```

---

## How It Works

### Step 1: Context is prepared with metadata

Before the LLM sees the retrieved passages, each one is wrapped with its source information:

```
[Passage 1 | Source: apple_10k_2023.pdf, Page: 52]  relevance: 2.341
The Company's net income for fiscal 2023 was $97.0 billion...

[Passage 2 | Source: apple_10k_2023.pdf, Page: 34]  relevance: 1.892
Research and development expenses were $29.9 billion...

[Passage 3 | Source: apple_10k_2023.pdf, Page: 18]  relevance: 1.234
Apple operates retail stores in 25 countries...
```

This makes it trivially easy for the LLM to cite — the `[Source: ...]` text is right there in the context.

### Step 2: The citation prompt enforces rules

The prompt (from `prompts/v1/shared.yaml`) gives the LLM strict rules:

```
CITATION RULES:
1. Every factual claim must be attributed using inline citations in the
   format [Source: <filename>, Page: <page>] or [Source: <filename>]
   if no page is available.
2. If multiple passages support a claim, cite all of them.
3. Quote exact key phrases from the source text when they are the best evidence.

DECLINE RULE:
4. If the context does NOT contain sufficient information to answer the question,
   respond with exactly:
   "I cannot answer this question based on the provided documents. The retrieved
   context does not contain sufficient information."
   Do NOT speculate, infer, or use outside knowledge.
```

### Step 3: Citations are extracted as structured data

The `RAGResponse` carries a `citations` field — a list of structured objects, not just free text:

```python
@dataclass
class RAGResponse:
    answer: str           # text with inline [Source: X, Page: Y] markers
    citations: list[dict] # structured: [{source, page, passage_preview, reranker_score}]
    sources: list[dict]   # backward-compatible source list
    ...
```

Example `citations` list:

```json
[
  {
    "source": "apple_10k_2023.pdf",
    "page": 52,
    "passage_preview": "The Company's net income for fiscal 2023 was $97.0 billion...",
    "reranker_score": 2.341
  },
  {
    "source": "apple_10k_2023.pdf",
    "page": 34,
    "passage_preview": "Research and development expenses were $29.9 billion...",
    "reranker_score": 1.892
  }
]
```

---

## Which Patterns Use Citation Enforcement?

Citation enforcement was rolled out to **all patterns** as part of the production enhancement:

| Pattern | Citation Style | Decline Rule |
|---|---|---|
| **Hybrid** | Full structured citations + reranker scores | ✅ Hard enforce |
| **Adaptive** | Structured citations on retrieval paths | ✅ Enforce |
| **Corrective (CRAG)** | Structured citations | ✅ Enforce |
| **HyDE** | Structured citations | ✅ Enforce |
| **Self-RAG** | Structured citations | ✅ Enforce |
| **Branched** | Structured citations | ✅ Enforce |
| **Graph** | Text + graph citations `[Graph: Entity]` | ✅ Enforce |
| **Agentic** | Inline citations enforced in system prompt | Partial |
| **Multimodal** | Source list (images + text) | Partial |

---

## API Response

Every response from the API now includes the `citations` field:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apples R&D spend in 2023?", "pattern": "hybrid"}'
```

```json
{
  "answer": "Apple spent $29.9 billion on research and development in fiscal year 2023 [Source: apple_10k_2023.pdf, Page: 34].",
  "pattern": "hybrid",
  "citations": [
    {
      "source": "apple_10k_2023.pdf",
      "page": 34,
      "passage_preview": "Research and development expenses were $29.9 billion for fiscal 2023...",
      "reranker_score": 2.841
    }
  ],
  "sources": [...],
  "metadata": {
    "declined": false,
    "prompt_version": "v1"
  }
}
```

### Detecting a declined answer

```python
response = requests.post("http://localhost:8000/query", json={...}).json()

if response["metadata"].get("declined"):
    print("Model couldn't answer — try rephrasing or ingesting more documents")
elif response["citations"]:
    for c in response["citations"]:
        print(f"Source: {c['source']}, Page {c['page']}")
        print(f"  {c['passage_preview'][:120]}...")
```

---

## UI Display

The Streamlit UI has a dedicated **Citations panel** in the query results:

```
┌─────────────────────────────────────────────────────────────┐
│  ANSWER                                                      │
│  Apple spent $29.9B on R&D in fiscal 2023                   │
│  [Source: apple_10k_2023.pdf, Page: 34].                    │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 🧠 Reasoning │  │ 📎 Sources   │  │ 🔖 Citations │  │ 🔧 Metadata  │
│              │  │              │  │ (2)          │  │              │
│ [expand]     │  │ [expand]     │  │              │  │ [expand]     │
│              │  │              │  │ [1] apple_   │  │              │
│              │  │              │  │ 10k_2023.pdf │  │              │
│              │  │              │  │ Page 34      │  │              │
│              │  │              │  │ rel: 2.841   │  │              │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

When the model declines, a yellow warning banner appears:

```
⚠️ Insufficient context — the model declined to answer because the retrieved
   documents do not contain sufficient information. Try ingesting more
   relevant documents or rephrasing your question.
```

---

## Customising the Citation Prompt

The citation rules live in `prompts/v1/shared.yaml`. You can make them stricter or more relaxed:

### Stricter version (v2 example)

```yaml
# prompts/v2/shared.yaml
citation_rag:
  system: |
    You are a legal document analyst. EVERY sentence must have a citation.
    You MUST quote the exact phrase from the source, not paraphrase.
    Format: "Exact quote from source" [Source: X, Page: Y]
    
    If ANY part of the answer is not supported, decline the entire answer.
```

### More permissive version

```yaml
# prompts/v2/shared.yaml  
citation_rag:
  system: |
    Answer the question using the provided context.
    Add a citation [Source: X, Page: Y] for key facts.
    For general statements, citations are optional.
```

Switch versions with:
```env
# .env
PROMPT_VERSION=v2
```

No code changes required. See [Prompt Versioning](02_prompt_versioning.md) for the full guide.

---

## Measuring Citation Quality

The RAGAS **faithfulness** metric directly measures whether the answer is grounded in context:

```python
from docustra.evaluation.metrics import evaluate_rag

result = evaluate_rag(
    questions=["What was Apple's R&D spend in 2023?"],
    answers=["Apple spent $29.9B on R&D [Source: apple_10k_2023.pdf, Page: 34]."],
    contexts=[["Research and development expenses were $29.9 billion..."]],
    ground_truths=["Apple's R&D spend was approximately $29.9 billion in fiscal 2023."]
)

print(f"Faithfulness: {result.faithfulness:.2f}")
# Faithfulness: 0.97 — answer is almost entirely grounded in context
```

A **faithfulness score of 1.0** means every claim in the answer is supported by the context. A score of 0.5 means half the claims are hallucinated. The CI gate requires ≥ 0.70.

---

## See Also

- [Hybrid RAG](../patterns/09_hybrid_rag.md) — Pattern with strongest citation integration
- [Prompt Versioning](02_prompt_versioning.md) — How to update the citation prompt
- [CI/CD Eval Gating](03_eval_ci_gating.md) — Measuring faithfulness in CI
