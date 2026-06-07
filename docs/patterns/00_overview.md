# RAG Patterns — Overview & Selection Guide

This document helps you choose the right RAG pattern for your use case.
Docustra implements **9 RAG patterns** — from lightweight routing to production-grade hybrid retrieval with cross-encoder reranking and mandatory citation enforcement.

---

## Pattern Comparison Matrix

| Pattern | Latency | Token Cost | Retrieval Quality | Citations | Best Use Case |
|---|---|---|---|---|---|
| [Adaptive](01_adaptive_rag.md) | Low–High | Low–Medium | Medium | ✅ Enforced | Mixed query workloads |
| [Agentic](02_agentic_rag.md) | High | High | Very High | Partial | Open-ended exploration |
| [Branched](03_branched_rag.md) | Medium | Medium | High | ✅ Enforced | Multi-faceted comparisons |
| [Corrective (CRAG)](04_corrective_rag.md) | Low–Medium | Low–Medium | High | ✅ Enforced | Uneven corpus coverage |
| [Graph](05_graph_rag.md) | Medium | Medium | Very High† | ✅ Enforced | Relationship questions |
| [HyDE](06_hyde_rag.md) | Low | Low | High | ✅ Enforced | Abstract/vague queries |
| [Multimodal](07_multimodal_rag.md) | Medium | Medium | High† | Partial | Image-rich documents |
| [Self-RAG](08_self_rag.md) | High | Very High | High | ✅ Enforced | Auditable, high-stakes answers |
| [**Hybrid** ⭐](09_hybrid_rag.md) | Medium | Low–Medium | **Highest** | ✅ **Strict** | Production "ask my docs" |

† Best in class for its specific domain  
⭐ Recommended for production use — combines BM25 + vector + cross-encoder reranking

---

## What's New: Production Enhancements

All patterns now include:

- **Citation enforcement** — every answer must cite source passages or decline. No more silent hallucination.
- **Prompt versioning** — all LLM prompts stored in `prompts/v1/*.yaml`, tunable without code changes.
- **Structured `citations[]` field** in every API response.

New pattern:

- **Hybrid RAG** — BM25 keyword search + dense vector search, merged with Reciprocal Rank Fusion, reranked by a cross-encoder. The highest-quality retrieval available. See [09_hybrid_rag.md](09_hybrid_rag.md).

---

## Decision Tree — Which Pattern to Use?

```
Is the question trivial (math, common knowledge)?
├── YES → Adaptive RAG (will skip retrieval automatically)
└── NO
    │
    Does the question reference images or charts?
    ├── YES → Multimodal RAG
    └── NO
        │
        Does the question ask about entity RELATIONSHIPS?
        ("How does X affect Y?", "What connects A and B?")
        ├── YES → Graph RAG
        └── NO
            │
            Is the question SHORT or ABSTRACT?
            ("What's Apple's strategy?", "Summarise the China risks")
            ├── YES → HyDE RAG
            └── NO
                │
                Is the answer HIGH-STAKES and must be AUDITABLE?
                (legal, compliance, medical decisions)
                ├── YES → Self-RAG
                └── NO
                    │
                    Does the question have MULTIPLE distinct dimensions?
                    ("Compare X, Y, and Z across FY2021-2023")
                    ├── YES → Branched RAG
                    └── NO
                        │
                        Is document coverage UNCERTAIN?
                        (query may not be in corpus)
                        ├── YES → Corrective RAG (CRAG)
                        └── NO
                            │
                            Does the question contain SPECIFIC TERMS?
                            (regulation names, clause numbers, proper nouns)
                            ├── YES → Hybrid RAG  ← best for technical/legal
                            └── NO
                                │
                                Does the question require ITERATIVE DISCOVERY?
                                (open-ended, exploratory)
                                ├── YES → Agentic RAG
                                └── NO → Hybrid RAG (best default for production)
```

**Bottom line:** When in doubt, use **Hybrid RAG** for production. Use **Adaptive RAG** when you want automatic complexity routing and can accept slightly lower retrieval quality.

---

## Latency vs. Quality Trade-off

```
Quality
  ▲
  │                                               ● Agentic
  │                                  ● Self-RAG
  │                       ● Hybrid ◄── NEW (best quality/latency balance)
  │                 ● Graph
  │          ● Branched
  │    ● CRAG
  │  ● HyDE
  │ ● Adaptive
  │● Standard RAG
  └────────────────────────────────────────────► Latency
  Fast                                        Slow
```

---

## All Patterns Now Return Citations

Every pattern now returns a structured `citations` list alongside the answer:

```json
{
  "answer": "Apple spent $29.9B on R&D [Source: apple_10k_2023.pdf, Page: 34].",
  "citations": [
    {
      "source": "apple_10k_2023.pdf",
      "page": 34,
      "passage_preview": "Research and development expenses were $29.9 billion...",
      "reranker_score": 2.341
    }
  ],
  "metadata": {
    "prompt_version": "v1",
    "declined": false
  }
}
```

When a pattern cannot find supporting context, it declines rather than hallucinating:

```json
{
  "answer": "I cannot answer this question based on the provided documents. The retrieved context does not contain sufficient information.",
  "citations": [],
  "metadata": { "declined": true }
}
```

---

## Composing Patterns

Patterns can be combined for maximum quality:

### Hybrid as the foundation

Hybrid RAG is the best general-purpose production pattern. You can stack other patterns on top:

```python
# Use Hybrid retrieval but with Self-RAG's reflection tokens for auditability
# → best of both worlds for high-stakes compliance use cases
```

### HyDE + CRAG (classic production combo)

Generate a hypothetical document → use it for retrieval → score relevance → correct if needed:

```python
# HyDE retrieval then CRAG validation
hypothetical = generate_hypothetical(question)
docs_with_scores = vector_store.similarity_search_with_scores(hypothetical)
if avg_score < threshold:
    fallback_to_web_search(question)
```

### Adaptive → Self-RAG (for compliance systems)

Route most queries through Adaptive, override to Self-RAG if marked high-stakes:

```python
if query.requires_audit:
    return SelfRAG().query(question)
return AdaptiveRAG().query(question)
```

### Graph + Branched (for deep research)

Decompose into branches, each branch uses Graph RAG for entity-aware retrieval:

```python
sub_questions = decompose(question)
for sq in sub_questions:
    entities = extract_entities(sq)
    graph_context = graph_store.get_entity_context(entities)
    # use graph_context + vector search per branch
```

---

## Docustra API — Using Each Pattern

All 9 patterns share the same endpoint. Switch by changing the `pattern` field:

```bash
# Hybrid (recommended for production — BM25 + vector + reranking + citations)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "pattern": "hybrid"}'

# Adaptive (default — routes by complexity)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "adaptive"}'

# Corrective (quality guard — scores and corrects retrieval)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "corrective"}'

# Self-RAG (auditable — reflection tokens logged)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "self_rag"}'

# Graph (entity relationships — uses Neo4j)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "graph"}'

# HyDE (abstract queries)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "hyde"}'

# Branched (multi-dimensional — parallel sub-questions)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "branched"}'

# Agentic (exploratory — LangGraph ReAct loop with tools)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "agentic"}'

# Multimodal (image-aware — pass file_path for vision)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "multimodal", "file_path": "data/report.pdf"}'
```

### Reading citations from the response

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What are Apple's main supply chain risks?",
    "pattern": "hybrid"
}).json()

print("Answer:", response["answer"])
print()

if response["metadata"].get("declined"):
    print("⚠ Model declined — insufficient context in documents")
else:
    print("Citations:")
    for i, c in enumerate(response["citations"], 1):
        print(f"  [{i}] {c['source']}, page {c['page']}")
        print(f"       {c['passage_preview'][:100]}...")
```

---

## Evaluating Each Pattern

Use the built-in CI evaluation script to benchmark patterns:

```bash
# Quick comparison: hybrid vs corrective vs adaptive
for pattern in hybrid corrective adaptive; do
    echo "=== $pattern ==="
    uv run python scripts/eval_ci.py --pattern $pattern --sample 15 --output ${pattern}_results.json
done
```

Or in Python:

```python
from docustra.evaluation.metrics import evaluate_rag
from docustra.retrieval import get_strategy, RAGPattern

question = "What are Apple's main risk factors?"
ground_truth = "Apple faces risks from supply chain concentration, regulatory scrutiny..."

results = {}
for pattern in [RAGPattern.HYBRID, RAGPattern.CORRECTIVE, RAGPattern.ADAPTIVE]:
    strategy = get_strategy(pattern)
    response = strategy.query(question)
    results[pattern.value] = evaluate_rag(
        questions=[question],
        answers=[response.answer],
        contexts=[[s.get("content", s.get("passage_preview", "")) for s in response.sources]],
        ground_truths=[ground_truth],
    ).as_dict()

for pattern, scores in results.items():
    print(f"{pattern:12} faithfulness={scores['faithfulness']:.2f}  "
          f"relevancy={scores['answer_relevancy']:.2f}  "
          f"precision={scores['context_precision']:.2f}")
```

---

## Pattern Deep-Dives

| # | Pattern | Documentation |
|---|---|---|
| 1 | Adaptive RAG | [01_adaptive_rag.md](01_adaptive_rag.md) |
| 2 | Agentic RAG | [02_agentic_rag.md](02_agentic_rag.md) |
| 3 | Branched RAG | [03_branched_rag.md](03_branched_rag.md) |
| 4 | Corrective RAG (CRAG) | [04_corrective_rag.md](04_corrective_rag.md) |
| 5 | Graph RAG | [05_graph_rag.md](05_graph_rag.md) |
| 6 | HyDE | [06_hyde_rag.md](06_hyde_rag.md) |
| 7 | Multimodal RAG | [07_multimodal_rag.md](07_multimodal_rag.md) |
| 8 | Self-RAG | [08_self_rag.md](08_self_rag.md) |
| 9 | **Hybrid RAG** ⭐ | [09_hybrid_rag.md](09_hybrid_rag.md) |

## Production Standards

| Feature | Documentation |
|---|---|
| Citation Enforcement | [production/01_citation_enforcement.md](../production/01_citation_enforcement.md) |
| Prompt Versioning | [production/02_prompt_versioning.md](../production/02_prompt_versioning.md) |
| CI/CD Eval Gating | [production/03_eval_ci_gating.md](../production/03_eval_ci_gating.md) |
