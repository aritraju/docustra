# Prompt Versioning

**Location:** `prompts/v1/` | **Config key:** `PROMPT_VERSION` | **Default:** `v1`

---

## Overview

Prompt versioning means storing LLM prompts in **files you can edit without touching Python code** — and keeping old versions so you can reproduce past results.

Before prompt versioning, every prompt looked like this:

```python
# src/docustra/retrieval/corrective.py  ← buried in code

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question to improve document retrieval."),
    ("human", "{question}"),
])
```

**Problems with this approach:**
- Changing a prompt requires editing Python source code
- Requires a developer — a domain expert can't tune the prompt
- When something breaks after a prompt change, you can't easily see what changed
- You can't A/B test two prompt versions side-by-side
- You can't reproduce a past run's exact behaviour

Prompt versioning solves all of these.

---

## The File Structure

```
docustra/
└── prompts/
    ├── v1/                          ← current active version
    │   ├── shared.yaml              ← prompts used by all patterns
    │   ├── adaptive.yaml            ← Adaptive RAG prompts
    │   ├── agentic.yaml             ← Agentic RAG system message
    │   ├── branched.yaml            ← Branched RAG prompts
    │   ├── corrective.yaml          ← CRAG prompts
    │   ├── graph.yaml               ← Graph RAG prompts
    │   ├── hybrid.yaml              ← Hybrid RAG metadata
    │   ├── hyde.yaml                ← HyDE prompts
    │   └── self_rag.yaml            ← Self-RAG reflection token prompts
    └── v2/                          ← your next version (not yet created)
        └── shared.yaml              ← only override what changed
```

---

## Anatomy of a Prompt File

Each YAML file contains named prompt **keys**. Each key has a `system` message (the instructions to the LLM) and optionally a `human` message (the user input template):

```yaml
# prompts/v1/corrective.yaml

rewrite_query:
  system: |
    Rewrite the question to improve document retrieval.
    Be more specific and use different keywords. Return only the rewritten question.
  human: "{question}"

relevance_score:
  system: |
    Score the relevance of this document to the question.
    Return ONLY a number between 0.0 and 1.0. Nothing else.
  human: "Question: {question}\n\nDocument: {document}"
```

### Template variables

Variables in `{curly_braces}` are filled in at runtime:

```yaml
# {question} gets replaced with the user's actual question
# {context} gets replaced with the retrieved document text
# {document} gets replaced with a single document's content
```

---

## How the Loader Works

The `get_prompt()` function in `src/docustra/core/prompts.py` handles everything:

```python
from docustra.core.prompts import get_prompt

# Load the "rewrite_query" prompt from "corrective.yaml" using the active version
prompt_template = get_prompt("corrective", "rewrite_query")

# Use it in a chain
chain = prompt_template | llm
result = chain.invoke({"question": "What are Apple's risks?"})
```

Under the hood:
1. Reads `PROMPT_VERSION` from settings (default: `"v1"`)
2. Opens `prompts/v1/corrective.yaml`
3. Finds the `rewrite_query` key
4. Builds a `ChatPromptTemplate` from the `system` and `human` fields
5. **Caches** the result (reads the file only once per version+module+key)

### What each parameter means

```python
get_prompt(
    module="corrective",   # YAML filename without extension
    key="rewrite_query",   # key within that file
    version=None           # None = use Settings.prompt_version
)
```

---

## The Active Version

Set the active version in your `.env` file:

```env
PROMPT_VERSION=v1
```

Every `RAGResponse` logs which version was used:

```json
{
  "metadata": {
    "prompt_version": "v1",
    ...
  }
}
```

This means you can look at any past response and know exactly which prompt produced it — critical for debugging and reproducibility.

---

## How to Create a New Prompt Version

### Step 1: Copy the current version

```bash
cp -r prompts/v1 prompts/v2
```

### Step 2: Edit only the prompts you want to change

```bash
# Edit the shared citation prompt to be stricter
nano prompts/v2/shared.yaml
```

```yaml
# prompts/v2/shared.yaml — stricter citations

citation_rag:
  system: |
    You are a legal document analyst. Follow these rules without exception:
    
    1. Every sentence in your answer must end with a citation:
       [Source: <filename>, Page: <page>]
    2. Quote exact phrases from the source — do not paraphrase.
    3. If any part of the answer cannot be cited, decline the entire answer with:
       "I cannot answer this question based on the provided documents."
    
    Context passages:
    {context}
  human: "{question}"
```

### Step 3: Test the new version

```bash
# Test just the new prompt without changing the production setting
uv run python scripts/eval_ci.py --prompt-version v2 --sample 10

# Or test interactively in Python
from docustra.core.prompts import get_prompt, invalidate_cache
import os
os.environ["PROMPT_VERSION"] = "v2"
invalidate_cache()

template = get_prompt("shared", "citation_rag")
print(template.format(context="test context", question="test question"))
```

### Step 4: Run the full evaluation gate

```bash
# Compare v1 vs v2
uv run python scripts/eval_ci.py --prompt-version v1 --output results_v1.json
uv run python scripts/eval_ci.py --prompt-version v2 --output results_v2.json

# Compare scores
python3 -c "
import json
v1 = json.load(open('results_v1.json'))['scores']
v2 = json.load(open('results_v2.json'))['scores']
for k in v1:
    diff = v2[k] - v1[k]
    arrow = '↑' if diff > 0 else '↓' if diff < 0 else '→'
    print(f'{k}: {v1[k]:.3f} → {v2[k]:.3f}  {arrow}{abs(diff):.3f}')
"
```

Expected output:
```
faithfulness:       0.72 → 0.84  ↑0.12
answer_relevancy:   0.78 → 0.76  ↓0.02
context_precision:  0.65 → 0.67  ↑0.02
```

### Step 5: Deploy

If v2 improves your target metrics:

```env
# .env
PROMPT_VERSION=v2
```

Restart the API — no code changes, no redeploy of packages.

---

## Prompt Files Reference

### `shared.yaml` — used by all patterns

```yaml
citation_rag:      # Answer generation with mandatory citations + decline rule
  variables: [context, question]
  
rag_basic:         # Simple answer without strict citation rules (legacy)
  variables: [context, question]
```

### `adaptive.yaml` — Adaptive RAG

```yaml
router:            # Classify question as trivial/simple/complex
  variables: [question]

direct_answer:     # Answer without retrieval (trivial path)
  variables: [question]

decompose:         # Break complex question into sub-questions
  variables: [question]

synthesize:        # Combine partial answers into final answer
  variables: [partial_answers, question]
```

### `corrective.yaml` — CRAG

```yaml
relevance_score:   # Score document relevance 0.0-1.0
  variables: [question, document]

rewrite_query:     # Improve query for better retrieval
  variables: [question]
```

### `self_rag.yaml` — Self-RAG reflection tokens

```yaml
retrieve_token:    # [Retrieve]: does this need documents? YES/NO
  variables: [question]

relevance_token:   # [Relevant]: is this doc relevant? YES/NO
  variables: [question, document]

support_token:     # [Supported]: is answer grounded? YES/PARTIALLY/NO
  variables: [context, answer]

useful_token:      # [Useful]: is this answer helpful? YES/NO
  variables: [question, answer]
```

### `branched.yaml` — Branched RAG

```yaml
decompose:         # Split into independent sub-questions
  variables: [question]

branch_answer:     # Answer one sub-question from its context
  variables: [context, question]

synthesize:        # Combine branch answers
  variables: [question, answers]
```

### `graph.yaml` — Graph RAG

```yaml
entity_extract:    # Extract named entities → JSON array
  variables: [question]

graph_answer:      # Answer using text + knowledge graph context
  variables: [question, text_context, graph_context]
```

### `hyde.yaml` — HyDE

```yaml
hypothetical_doc:  # Generate hypothetical document for embedding
  variables: [question]
```

---

## Edge Cases and Error Handling

### What if a key doesn't exist?

```python
from docustra.core.prompts import get_prompt

try:
    prompt = get_prompt("corrective", "nonexistent_key")
except KeyError as e:
    print(e)
# KeyError: "Prompt key 'nonexistent_key' not found in corrective.yaml (v1).
#            Available keys: ['relevance_score', 'rewrite_query']"
```

### What if the YAML file is malformed?

The CI workflow has a dedicated step that validates all YAML files:

```bash
python3 -c "
import yaml, sys
from pathlib import Path
errors = []
for f in Path('prompts').rglob('*.yaml'):
    try:
        yaml.safe_load(open(f))
        print(f'✓ {f}')
    except yaml.YAMLError as e:
        errors.append(str(e))
if errors:
    sys.exit(1)
"
```

This runs as the `lint-prompts` job in `.github/workflows/eval.yml` on every pull request.

### Clearing the cache during testing

If you change a YAML file while a Python process is running, call `invalidate_cache()`:

```python
from docustra.core.prompts import invalidate_cache
invalidate_cache()  # forces next get_prompt() call to re-read files
```

---

## Multi-line YAML Tips

YAML `|` (literal block) preserves newlines — use it for long prompts:

```yaml
my_prompt:
  system: |
    Line one.
    Line two.
    Line three.
```

YAML `>` (folded block) collapses newlines into spaces — use it for prose:

```yaml
my_prompt:
  system: >
    This is a single paragraph that
    happens to be written across
    multiple lines in the file.
```

Always use `|` for prompts that have numbered lists or structured formatting.

---

## See Also

- [Citation Enforcement](01_citation_enforcement.md) — How the `citation_rag` prompt works
- [CI/CD Eval Gating](03_eval_ci_gating.md) — Testing prompt changes before merge
- [Hybrid RAG](../patterns/09_hybrid_rag.md) — Pattern that uses both shared and hybrid prompts
