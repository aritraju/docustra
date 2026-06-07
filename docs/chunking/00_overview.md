# Chunking Strategies — Overview

Chunking is the process of splitting documents into smaller pieces before embedding and storing them in Qdrant. The chunking strategy you choose directly affects retrieval quality — wrong chunk boundaries mean relevant information gets split across chunks, missed by similarity search, or returned without enough context for the LLM to answer correctly.

Docustra supports **10 chunking strategies** selectable per ingestion, from the sidebar in the UI or via the API.

---

## Quick Comparison

| Strategy | Speed | Chunk Quality | LLM Required | Best For |
|---|---|---|---|---|
| [Recursive](01_recursive.md) | ⚡ Fast | Good | No | General purpose (default) |
| [Character](02_character.md) | ⚡ Fast | Basic | No | Structured paragraph text |
| [Token](03_token.md) | ⚡ Fast | Good | No | LLM context window compliance |
| [Sentence Transformers](04_sentence_transformers.md) | ⚡ Fast | Good | No | Preventing embedding truncation |
| [Semantic](05_semantic.md) | 🐢 Slow* | Excellent | No** | Technical docs, topic shifts |
| [Sentence Window](06_sentence_window.md) | ⚡ Fast | Excellent | No | Dense text, precise retrieval |
| [Markdown](07_markdown.md) | ⚡ Fast | Very Good | No | Markdown docs, wikis, READMEs |
| [HTML](08_html.md) | ⚡ Fast | Very Good | No | Web pages, HTML reports |
| [Parent-Child](09_parent_child.md) | ⚡ Fast | Excellent | No | Long documents, best accuracy |
| [Hypothetical Questions](10_hypothetical_questions.md) | 🐢 Slow | Excellent | Yes 🤖 | Q&A, conversational docs |

*Slow because it embeds every sentence to detect topic changes
**Uses the embedding model, not the LLM

---

## How to Select a Strategy

**Via UI:** Sidebar → Chunking Strategy dropdown → select → Ingest

**Via API:**
```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@document.pdf" \
  -F "chunking_strategy=semantic"
```

**List all available strategies:**
```bash
curl http://localhost:8000/ingest/strategies
```

---

## Decision Guide

```
What type of document are you ingesting?
│
├── Markdown file (.md) → Markdown strategy
├── HTML / web page     → HTML strategy
├── Source code         → Recursive (code mode)
│
└── PDF / plain text
      │
      ├── Is it FAQ-style / Q&A format?
      │     └── YES → Hypothetical Questions
      │
      ├── Is dense technical text with long passages?
      │     └── YES → Sentence Window or Parent-Child
      │
      ├── Do you need maximum retrieval accuracy?
      │     └── YES → Parent-Child
      │
      ├── Does topic change frequently within sections?
      │     └── YES → Semantic
      │
      ├── Using GPT-4 / Claude with strict token limits?
      │     └── YES → Token
      │
      └── General purpose
            └── Recursive (default)
```

---

## Chunk Size Configuration

All non-LLM strategies respect the `CHUNK_SIZE` and `CHUNK_OVERLAP` settings in `.env`:

```env
CHUNK_SIZE=650     # characters (updated default — midpoint of 500-800 token sweet spot)
CHUNK_OVERLAP=100  # overlap between consecutive chunks (updated — preserves cross-boundary context)
```

### Why 650 / 100?

The **500–800 token range** is the empirical sweet spot for RAG chunk sizes — established through benchmarks on passage retrieval tasks:

- **Too small (< 300 tokens):** chunks lose context, the LLM gets fragments instead of coherent passages
- **Too large (> 1000 tokens):** chunks are too general, retrieval precision drops, and the LLM has to filter noise
- **650 tokens:** midpoint of the sweet spot — balances context richness with retrieval precision

The **100-token overlap** (up from 64) ensures that sentences spanning chunk boundaries are captured in at least one chunk, which is critical for questions that reference specific numbers or facts at paragraph edges.

### How chunk size interacts with retrieval

```
Small chunks (256 tokens):
  Retrieved 5 chunks → 5 × 256 = 1,280 tokens of context
  ✓ Very precise — each chunk is tightly focused
  ✗ May miss multi-sentence reasoning
  
Default chunks (650 tokens):
  Retrieved 5 chunks → 5 × 650 = 3,250 tokens of context
  ✓ Each chunk tells a coherent story
  ✓ Enough context for multi-sentence answers
  ✓ Balanced with typical LLM context windows
  
Large chunks (1024 tokens):
  Retrieved 5 chunks → 5 × 1,024 = 5,120 tokens of context
  ✓ Maximum context per chunk
  ✗ May dilute relevance score in hybrid/reranking
```

### Tuning guidance

| Document type | Recommended `CHUNK_SIZE` | Recommended `CHUNK_OVERLAP` |
|---|---|---|
| Dense technical (research papers) | 800–1024 | 100–150 |
| General enterprise docs | 650 (default) | 100 (default) |
| Legal / regulatory (long clauses) | 800–1200 | 150–200 |
| FAQ / Q&A format | 300–400 | 50–75 |
| Short news articles / briefs | 400–500 | 75–100 |

---

## What Happens After Chunking

Every chunk becomes a `Document` object with:
- `page_content` — the text that gets embedded
- `metadata` — source file, page number, chunk type, strategy-specific fields

These are then:
1. **Embedded** by `sentence-transformers/all-MiniLM-L6-v2` (locally on MPS)
2. **Stored** in Qdrant as a 384-dimensional vector + payload
3. **Retrieved** at query time via cosine similarity search

---

## Detailed Documentation

| # | Strategy | Doc |
|---|---|---|
| 1 | Recursive Character | [01_recursive.md](01_recursive.md) |
| 2 | Character | [02_character.md](02_character.md) |
| 3 | Token | [03_token.md](03_token.md) |
| 4 | Sentence Transformers Token | [04_sentence_transformers.md](04_sentence_transformers.md) |
| 5 | Semantic | [05_semantic.md](05_semantic.md) |
| 6 | Sentence Window | [06_sentence_window.md](06_sentence_window.md) |
| 7 | Markdown | [07_markdown.md](07_markdown.md) |
| 8 | HTML | [08_html.md](08_html.md) |
| 9 | Parent-Child | [09_parent_child.md](09_parent_child.md) |
| 10 | Hypothetical Questions | [10_hypothetical_questions.md](10_hypothetical_questions.md) |
