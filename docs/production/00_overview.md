# Production Standards — Overview

This section covers the features that separate a **demo RAG system** from a **production RAG system**. These are not optional additions — they are the difference between a system you can trust and one that might quietly give wrong answers.

---

## What Makes RAG "Production-Grade"?

A basic RAG system can be built in an afternoon:
1. Split document into chunks
2. Embed and store in a vector database
3. Retrieve top-K chunks at query time
4. Pass to LLM and return answer

A **production** RAG system additionally guarantees:

| Concern | Basic RAG | Production RAG |
|---|---|---|
| **Answer accuracy** | LLM may hallucinate | Citations enforce grounding |
| **Retrieval quality** | Single vector search | Hybrid BM25 + vector + reranking |
| **Prompt changes** | Edit code, redeploy | Versioned YAML files, zero-code changes |
| **Quality regression** | Discovered by users | Caught by automated CI evaluation gate |
| **Audit trail** | None | Every answer has cited passages + metadata |

---

## Production Features in Docustra

### [01 — Citation Enforcement](01_citation_enforcement.md)

Every answer must cite the exact source passage or explicitly decline to answer. This prevents hallucination and makes answers reviewable.

**Key benefit:** Users can verify every claim. Compliance teams can audit answers.

```
Answer: "Apple spent $29.9B on R&D in fiscal 2023 [Source: apple_10k_2023.pdf, Page: 34]."
```

### [02 — Prompt Versioning](02_prompt_versioning.md)

All LLM prompts are stored in versioned YAML files (`prompts/v1/`), not hardcoded in Python. Changing a prompt is a YAML edit, not a code deployment.

**Key benefit:** Non-engineers can tune prompts. Old behaviour is reproducible by version ID.

```yaml
# prompts/v1/shared.yaml
citation_rag:
  system: "Answer using ONLY the provided context. Cite every claim..."
```

### [03 — CI/CD Evaluation Gating](03_eval_ci_gating.md)

A golden dataset of 50 QA pairs runs RAGAS evaluation on every pull request. If faithfulness drops below 0.70, the build fails — catching retrieval regressions before they reach users.

**Key benefit:** Prompt or retrieval changes can never silently degrade answer quality.

```
✗ faithfulness: 0.61 < threshold 0.70
⚠ Build blocked until scores improve.
```

---

## How the Three Features Work Together

```
Developer changes a prompt
         │
         ▼
  Edits prompts/v2/shared.yaml
         │
         ▼
  Opens a pull request
         │
         ▼
  GitHub Actions runs eval_ci.py
  against golden_dataset.json
         │
    ┌────▼────┐
    │ PASSES? │
    └────┬────┘
         │
    YES  │  NO
         │   │
         ▼   ▼
      Merge  Block PR
      ✓      ✗
             │
             ▼
       Developer sees:
       "faithfulness: 0.61
        < threshold 0.70"
         
       Fixes prompt in YAML,
       re-opens PR
```

---

## Quick Reference

| Feature | Config Key | Default | File |
|---|---|---|---|
| Prompt version | `PROMPT_VERSION` | `v1` | `.env` |
| Faithfulness threshold | `EVAL_FAITHFULNESS_THRESHOLD` | `0.70` | `.env` |
| Answer relevancy threshold | `EVAL_ANSWER_RELEVANCY_THRESHOLD` | `0.70` | `.env` |
| Context precision threshold | `EVAL_CONTEXT_PRECISION_THRESHOLD` | `0.60` | `.env` |
| BM25 weight | `BM25_WEIGHT` | `0.4` | `.env` |
| Reranker model | `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `.env` |
| Enable reranking | `ENABLE_RERANKING` | `true` | `.env` |

---

## Running the Evaluation Gate Manually

```bash
# Quick check — 10 pairs, hybrid pattern
uv run python scripts/eval_ci.py --sample 10 --pattern hybrid

# Full evaluation — all 50 pairs
uv run python scripts/eval_ci.py

# Domain-specific
uv run python scripts/eval_ci.py --domain apple_10k --sample 25

# Save results to file
uv run python scripts/eval_ci.py --output results.json
```

---

## Detailed Guides

| Guide | What It Covers |
|---|---|
| [Citation Enforcement](01_citation_enforcement.md) | How citations work, the decline rule, UI display |
| [Prompt Versioning](02_prompt_versioning.md) | YAML structure, the prompt loader, how to create v2 |
| [CI/CD Eval Gating](03_eval_ci_gating.md) | Golden dataset, RAGAS metrics, GitHub Actions workflow |
