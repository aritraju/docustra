# RAG Patterns — Overview & Selection Guide

This document helps you choose the right RAG pattern for your use case.

---

## Pattern Comparison Matrix

| Pattern | Latency | Token Cost | Retrieval Quality | Auditability | Best Use Case |
|---|---|---|---|---|---|
| [Adaptive](01_adaptive_rag.md) | Low-High | Low-Medium | Medium | Low | Mixed query workloads |
| [Agentic](02_agentic_rag.md) | High | High | Very High | Medium | Open-ended exploration |
| [Branched](03_branched_rag.md) | Medium | Medium | High | Medium | Multi-faceted comparisons |
| [Corrective (CRAG)](04_corrective_rag.md) | Low-Medium | Low-Medium | High | Medium | Uneven corpus coverage |
| [Graph](05_graph_rag.md) | Medium | Medium | Very High* | Medium | Relationship questions |
| [HyDE](06_hyde_rag.md) | Low | Low | High | Low | Abstract/vague queries |
| [Multimodal](07_multimodal_rag.md) | Medium | Medium | High* | Low | Image-rich documents |
| [Self-RAG](08_self_rag.md) | High | Very High | High | Very High | High-stakes answers |

*Best in class for its specific problem domain

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
            ("What's Apple's strategy?", "China risks")
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
                            Does the question require ITERATIVE DISCOVERY?
                            (open-ended, you don't know what to search for)
                            ├── YES → Agentic RAG
                            └── NO → Standard RAG (Adaptive simple path)
```

---

## Latency vs. Quality Trade-off

```
Quality
  ▲
  │                                           ● Agentic
  │                              ● Self-RAG
  │                   ● Graph
  │          ● Branched
  │    ● CRAG
  │  ● HyDE
  │ ● Adaptive
  │● Standard RAG
  └────────────────────────────────────────► Latency
  Fast                                    Slow
```

---

## Composing Patterns

Patterns can be combined for maximum quality:

### HyDE + CRAG (Recommended for production)
Generate a hypothetical document → use it for retrieval → score relevance → correct if needed:
```python
# HyDE retrieval, then CRAG validation
hypothetical = generate_hypothetical(question)
docs_with_scores = vector_store.similarity_search_with_scores(hypothetical)
if avg_score < threshold:
    fallback_to_web_search(question)
```

### Adaptive → Self-RAG (For compliance systems)
Route all queries through Adaptive, but override to Self-RAG if marked high-stakes:
```python
if query.requires_audit:
    return SelfRAG().query(question)
return AdaptiveRAG().query(question)
```

### Graph + Branched (For deep research)
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

All patterns share the same endpoint. Switch patterns by changing the `pattern` field:

```bash
# Adaptive (default — routes automatically)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "adaptive"}'

# Agentic (best for exploration)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "agentic"}'

# Graph (entity relationship questions)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "graph"}'

# HyDE (abstract queries)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "hyde"}'

# CRAG (quality guarantee)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "corrective"}'

# Self-RAG (auditable answers)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "self_rag"}'

# Branched (multi-dimensional)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "branched"}'

# Multimodal (image-aware)
curl -X POST http://localhost:8000/query \
  -d '{"question": "...", "pattern": "multimodal", "file_path": "data/report.pdf"}'
```

---

## Evaluating Each Pattern

Use RAGAS to benchmark patterns head-to-head on the same question set:

```python
# src/docustra/evaluation/metrics.py
from docustra.evaluation.metrics import evaluate_rag
from docustra.retrieval import get_strategy, RAGPattern

question = "What are Apple's main risk factors?"
ground_truth = "Apple's main risks include supply chain concentration..."

results = {}
for pattern in RAGPattern:
    strategy = get_strategy(pattern)
    response = strategy.query(question)
    results[pattern.value] = evaluate_rag(
        questions=[question],
        answers=[response.answer],
        contexts=[[s["content"] for s in response.sources]],
        ground_truths=[ground_truth],
    ).as_dict()

# Compare patterns
for pattern, scores in results.items():
    print(f"{pattern}: faithfulness={scores['faithfulness']:.2f}, "
          f"relevancy={scores['answer_relevancy']:.2f}")
```

---

## Pattern Deep-Dives

| Pattern | Documentation |
|---|---|
| Adaptive RAG | [01_adaptive_rag.md](01_adaptive_rag.md) |
| Agentic RAG | [02_agentic_rag.md](02_agentic_rag.md) |
| Branched RAG | [03_branched_rag.md](03_branched_rag.md) |
| Corrective RAG | [04_corrective_rag.md](04_corrective_rag.md) |
| Graph RAG | [05_graph_rag.md](05_graph_rag.md) |
| HyDE | [06_hyde_rag.md](06_hyde_rag.md) |
| Multimodal RAG | [07_multimodal_rag.md](07_multimodal_rag.md) |
| Self-RAG | [08_self_rag.md](08_self_rag.md) |
