# CI/CD Evaluation Gating

**Script:** `scripts/eval_ci.py` | **Workflow:** `.github/workflows/eval.yml` | **Dataset:** `data/eval/golden_dataset.json`

---

## Overview

CI/CD evaluation gating means **your RAG pipeline is automatically tested on every code change**, and the build fails if answer quality drops below defined thresholds.

Think of it as unit tests for your AI system. Except instead of testing "does this function return the right value?", you're testing "does this system give accurate, grounded, relevant answers?"

---

## Why This Matters

Consider what happens without evaluation gating:

```
Developer: "I improved the retrieval prompt — seems to work better on my test questions."
               ↓
         Merges PR
               ↓
         Deploys to production
               ↓
  Users start getting vague, uncited answers
               ↓
  Support tickets filed 3 days later
               ↓
  Developer has to figure out which of 12 recent changes caused it
```

With evaluation gating:

```
Developer changes a prompt
               ↓
  Opens a pull request
               ↓
  GitHub Actions runs 25 QA pairs through the system
               ↓
  faithfulness: 0.61 < threshold 0.70
  ✗ Build BLOCKED
               ↓
  Developer sees the failure immediately, in the PR
               ↓
  Fixes the prompt before merging
```

Quality regressions are caught in minutes, not days.

---

## The Golden Dataset

The golden dataset lives at `data/eval/golden_dataset.json`. It contains **50 QA pairs** across two document domains:

```json
{
  "_meta": {
    "version": "v1",
    "total_pairs": 50,
    "documents": ["apple_10k_2023.pdf", "vector-databses-guide.pdf"]
  },
  "pairs": [
    {
      "id": "apple_001",
      "domain": "apple_10k",
      "question": "What was Apple's total net sales for fiscal year 2023?",
      "ground_truth": "Apple's total net sales for fiscal year 2023 were $383.3 billion...",
      "tags": ["financials", "revenue"]
    },
    {
      "id": "vdb_010",
      "domain": "vector_databases",
      "question": "What is hybrid search in the context of vector databases?",
      "ground_truth": "Hybrid search combines dense vector search with sparse keyword search...",
      "tags": ["hybrid_search", "BM25"]
    }
    ...
  ]
}
```

### Domain breakdown

| Domain | Pairs | Topics |
|---|---|---|
| `apple_10k` | 25 | Financials, risk factors, products, sustainability, legal |
| `vector_databases` | 25 | Indexing algorithms, similarity metrics, RAG concepts |

### What makes a good golden dataset?

A good golden dataset has:
1. **Representative coverage** — questions that span the range of real user queries
2. **Clear ground truths** — specific, verifiable correct answers
3. **Diverse difficulty** — some easy lookups, some multi-sentence reasoning
4. **Edge cases** — questions where the document coverage is thin, to test the decline rule
5. **Stable answers** — facts that don't change (not "Apple's current stock price")

---

## The Four RAGAS Metrics

RAGAS evaluates four dimensions of RAG quality. All four are measured in Docustra's evaluation.

### 1. Faithfulness

> *Are the claims in the answer supported by the retrieved context?*

This is the most important metric for citation enforcement. A faithfulness score of 1.0 means every claim in the answer appears in the retrieved passages. A score of 0.0 means the answer is entirely hallucinated.

```
Question: "What was Apple's R&D spend?"
Context: "R&D expenses were $29.9 billion in fiscal 2023."
Answer: "Apple spent $29.9 billion on R&D."  → Faithfulness: 1.0 ✓

Answer: "Apple spent $35 billion on R&D."   → Faithfulness: 0.0 ✗
                                                (claim not in context)
```

**Threshold:** 0.70 — at least 70% of claims must be traceable to context.

### 2. Answer Relevancy

> *Does the answer actually address the question that was asked?*

A perfectly faithful answer that doesn't answer the question is still a bad answer. This metric checks alignment between the question and the answer.

```
Question: "What are Apple's supply chain risks?"
Answer: "Apple is headquartered in Cupertino, California."
→ Answer Relevancy: 0.05  ✗  (faithful but irrelevant)

Answer: "Apple relies on single-source suppliers in Asia [Source: ...]"
→ Answer Relevancy: 0.95  ✓
```

**Threshold:** 0.70 — the answer must address the question.

### 3. Context Precision

> *Were the retrieved chunks actually useful for answering?*

This measures whether the retrieved passages contain information relevant to the question. A low score means your retrieval is bringing in irrelevant documents that the LLM has to ignore.

```
Question: "What was Apple's R&D spend?"
Retrieved: [doc about R&D expenses, doc about competitor products, doc about store locations]
→ Context Precision: 0.33  ✗  (only 1 of 3 retrieved docs was relevant)

Retrieved: [doc about R&D expenses, doc about R&D team size, doc about R&D strategy]
→ Context Precision: 1.0   ✓
```

**Threshold:** 0.60 — at least 60% of retrieved chunks should be relevant.

### 4. Context Recall

> *Did the retrieved context cover everything needed for the full answer?*

This requires a `ground_truth` answer to compare against. It measures whether the retrieved passages contain all the information present in the ground truth.

**Note:** This metric is computed when `ground_truth` is provided in the golden dataset. It is reported but does not currently gate the build (no threshold), because it requires highly specific ground truths.

---

## Running the Evaluation Gate

### Quick run (development)

```bash
# 10 pairs, hybrid pattern — fast sanity check
uv run python scripts/eval_ci.py --sample 10 --pattern hybrid
```

### Full run (before merging)

```bash
uv run python scripts/eval_ci.py
```

### Domain-specific run

```bash
# Only Apple 10-K questions
uv run python scripts/eval_ci.py --domain apple_10k --sample 25

# Only vector database questions
uv run python scripts/eval_ci.py --domain vector_databases --sample 25
```

### Compare two patterns

```bash
# Test hybrid vs corrective
uv run python scripts/eval_ci.py --pattern hybrid --output hybrid_results.json
uv run python scripts/eval_ci.py --pattern corrective --output corrective_results.json

python3 -c "
import json
for name in ['hybrid', 'corrective']:
    d = json.load(open(f'{name}_results.json'))
    print(f'\\n{name}:')
    for k, v in d['scores'].items():
        print(f'  {k}: {v:.4f}')
"
```

### Test a new prompt version before merging

```bash
uv run python scripts/eval_ci.py --prompt-version v2 --sample 25 --output v2_results.json
```

### Example output

```
============================================================
  DOCUSTRA RAG EVALUATION RESULTS
============================================================
  Pattern:        hybrid
  Prompt version: v1
  Pairs evaluated:25

  Scores:
    ✓ faithfulness              0.8234  (threshold: 0.70)
    ✓ answer_relevancy          0.7891  (threshold: 0.70)
    ✓ context_precision         0.6723  (threshold: 0.60)
    → context_recall            0.7156

  ✓ All thresholds passed — build gates cleared.
============================================================
```

### Failed run output

```
============================================================
  DOCUSTRA RAG EVALUATION RESULTS
============================================================
  Pattern:        adaptive
  Prompt version: v2
  Pairs evaluated:25

  Scores:
    ✗ faithfulness              0.6134  (threshold: 0.70)
    ✓ answer_relevancy          0.7512  (threshold: 0.70)
    ✓ context_precision         0.6234  (threshold: 0.60)

  FAILED — thresholds not met:
    ✗ faithfulness: 0.6134 < threshold 0.70

  ⚠ Build should be blocked until scores improve.
    See docs/eval-improvement.md for guidance.
============================================================

Exit code: 1  ← CI picks this up and fails the build
```

---

## The GitHub Actions Workflow

The workflow file lives at `.github/workflows/eval.yml`.

### When it runs

The workflow triggers on:
- Pushes to `main` or `develop` branches **that touch retrieval or prompt files**
- Pull requests to `main`
- Manual trigger (`workflow_dispatch`) — useful for testing

```yaml
on:
  push:
    paths:
      - "src/docustra/retrieval/**"
      - "prompts/**"
      - "data/eval/**"
      - "scripts/eval_ci.py"
  pull_request:
    branches: [main]
```

### What it does

```
GitHub Actions triggers
        │
        ▼
  Start Qdrant in Docker (service container)
        │
        ▼
  Install Python 3.11 + uv
        │
        ▼
  uv sync (install all dependencies)
        │
        ▼
  Wait for Qdrant to be healthy
        │
        ▼
  Ingest sample documents into Qdrant
        │
        ▼
  Run eval_ci.py for each pattern in matrix:
    ├── hybrid
    ├── corrective
    └── adaptive
        │
        ▼
  Upload eval_results_*.json as artifacts
        │
        ▼
  Post score summary to GitHub PR page
        │
    ┌───▼───┐
    │ PASS? │
    └───┬───┘
     YES│  NO
        │   │
      ✓ │   │ ✗
        │   │
   Allow  Block
   merge   PR
```

### The pattern matrix

The workflow tests three patterns in parallel using GitHub's matrix strategy:

```yaml
strategy:
  matrix:
    pattern: [hybrid, corrective, adaptive]
```

This means three separate evaluation jobs run simultaneously — faster CI and coverage of multiple patterns.

### Reading results in GitHub

After a run, you'll see a summary on the PR page like:

```markdown
## RAG Evaluation: hybrid
{
  "scores": {
    "faithfulness": 0.8234,
    "answer_relevancy": 0.7891,
    "context_precision": 0.6723
  },
  "passed": true
}
```

And individual result JSON files are available as downloadable **build artifacts** for 30 days.

---

## Configuring Thresholds

Thresholds are set in `.env` and read by `Settings`:

```env
# .env
EVAL_FAITHFULNESS_THRESHOLD=0.70
EVAL_ANSWER_RELEVANCY_THRESHOLD=0.70
EVAL_CONTEXT_PRECISION_THRESHOLD=0.60
```

They are also configurable as GitHub Actions environment variables for the CI environment:

```yaml
# .github/workflows/eval.yml
env:
  EVAL_FAITHFULNESS_THRESHOLD: "0.70"
  EVAL_ANSWER_RELEVANCY_THRESHOLD: "0.70"
  EVAL_CONTEXT_PRECISION_THRESHOLD: "0.60"
```

### Choosing threshold values

| Threshold | What it means | When to raise |
|---|---|---|
| **0.50** | Baseline — more than half of claims grounded | Low-stakes internal tools |
| **0.70** | Default — strong production quality | Most enterprise RAG |
| **0.85** | High quality — near-zero hallucination rate | Medical, legal, compliance |
| **0.95** | Very strict — essentially no unsupported claims | Safety-critical applications |

Start with the defaults (0.70 faithfulness, 0.70 relevancy). Once you know your system's baseline, adjust based on your use case.

---

## Maintaining the Golden Dataset

The golden dataset should evolve with your documents.

### Adding new QA pairs

```json
{
  "id": "apple_026",
  "domain": "apple_10k",
  "question": "What new products did Apple announce in 2023?",
  "ground_truth": "Apple introduced the iPhone 15 series and Apple Vision Pro...",
  "tags": ["products", "new_launches"]
}
```

### Guidelines for good QA pairs

**Do:**
- Use questions that real users would ask
- Write ground truths that are specific and verifiable
- Include questions that test the decline rule (no answer in corpus)
- Tag questions by topic so you can run domain-specific tests

**Don't:**
- Write questions with ambiguous correct answers
- Use time-sensitive facts ("current stock price")
- Make ground truths so long they're hard to match
- Include only easy questions — include hard ones too

### Version the dataset

When you add many new pairs or significantly change ground truths, bump the dataset version in `_meta`:

```json
{
  "_meta": {
    "version": "v2",
    "total_pairs": 100,
    "description": "Added 50 pairs for new product documentation domain"
  }
}
```

---

## Troubleshooting Common Failures

### Low faithfulness (< 0.70)

The model is making claims not in the retrieved context.

**Diagnosis:**
```bash
# Look at which questions are failing
uv run python scripts/eval_ci.py --sample 50 --output results.json
python3 -c "
import json
d = json.load(open('results.json'))
for q, a in zip(d['sample_responses'][:5], []):
    print(q['question'])
    print(q['answer'][:200])
    print()
"
```

**Common causes and fixes:**
- The answer prompt is too permissive → tighten the citation rules in `prompts/v1/shared.yaml`
- Retrieved chunks are too small → increase `CHUNK_SIZE` in `.env`
- Wrong documents ingested → re-ingest with correct files

### Low answer relevancy (< 0.70)

The model answers something other than what was asked.

**Common causes and fixes:**
- Retrieval is returning unrelated chunks → increase `RERANKER_TOP_N` or tune `BM25_WEIGHT`
- The question decomposition in Adaptive/Branched is off → tune the `decompose` prompt

### Low context precision (< 0.60)

Too many irrelevant chunks are being retrieved.

**Common causes and fixes:**
- `RETRIEVAL_TOP_K` is too high → reduce it (e.g., from 10 to 5)
- Chunk size is too large (overly general chunks) → reduce `CHUNK_SIZE`
- BM25 weight too low → increase `BM25_WEIGHT` to improve keyword matching

---

## See Also

- [Citation Enforcement](01_citation_enforcement.md) — What faithfulness measures
- [Prompt Versioning](02_prompt_versioning.md) — How to safely iterate on prompts
- [Hybrid RAG](../patterns/09_hybrid_rag.md) — The default pattern for evaluation
- [RAGAS documentation](https://docs.ragas.io) — Full metric explanations
