# HyDE — Hypothetical Document Embedding

## Overview

HyDE (Hypothetical Document Embedding) solves a fundamental **embedding space mismatch** between user queries and stored documents. A user query is typically short, conversational, and abstract. Documents are long, formal, and specific. When embedded, these two representations live in very different parts of the vector space — even when they're semantically related.

HyDE bridges this gap by generating a **hypothetical document** that answers the query, and using *that document's embedding* for retrieval instead of the query's embedding.

---

## The Core Insight

```
Standard RAG embedding space:

  "What drove Apple's services growth?"   ← short, abstract query
                 ↓ embed
           [0.23, -0.41, 0.18, ...]     ← query vector
                 
           DISTANCE
           
  "Services net revenues increased to $85.2 billion    ← long, formal passage
   in 2023 from $78.1 billion in 2022, primarily due
   to growth in the App Store, Apple Music..."
                 ↓ embed
           [0.31, -0.17, 0.44, ...]     ← document vector
           
   Cosine similarity: 0.52 (moderate)
```

```
HyDE embedding space:

  LLM generates hypothetical:
  "Apple's services revenue grew significantly in 2023,
   driven by strong App Store performance and increased
   Apple Music subscribers..."
                 ↓ embed
           [0.29, -0.14, 0.41, ...]     ← hypothetical doc vector
                 
   vs. actual document:
           [0.31, -0.17, 0.44, ...]     ← document vector
           
   Cosine similarity: 0.84 (high) ✅
```

The hypothetical document shares vocabulary, style, and structure with real documents — making it a much better search anchor.

---

## Architecture

```
        User Query (short, abstract)
               │
               ▼
┌──────────────────────────────────────────┐
│          LLM — Hypothetical Generator     │
│                                          │
│  Prompt: "Write a document passage that  │
│   would answer this question. Write it   │
│   as a formal enterprise document."      │
│                                          │
│  Output: 2-3 sentence formal passage     │
│          ← looks like a real document    │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  embed(hypothetical_doc)                  │
│  ← uses same embedding model as corpus   │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  VectorStore.similarity_search(           │
│      hypothetical_doc_text, k=5          │  ← search with hypothetical text
│  )                                        │
│                                          │
│  Returns: REAL documents from corpus     │  ← NOT the hypothetical
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  LLM — Answer Generator                  │
│                                          │
│  Input: real retrieved documents         │  ← real docs, not hypothetical
│  Output: final grounded answer           │
└──────────────────────────────────────────┘
```

**Critical distinction:** The hypothetical document is ONLY used as the search vector. The final answer is generated from *real* retrieved documents. This ensures factual grounding.

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/hyde.py`

### Step 1 — Generate Hypothetical Document

```python
_HYPOTHETICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Write a hypothetical document passage that would answer the following question.
Write it as if it were an excerpt from a formal enterprise document or report.
Be specific and factual in tone. Length: 2-3 sentences."""),
    ("human", "{question}"),
])

chain = _HYPOTHETICAL_PROMPT | self._llm
hypothetical_doc = chain.invoke({"question": question}).content.strip()
```

The prompt instructs the LLM to write *as if from a formal document* — this is key to matching the embedding style of the corpus.

### Step 2 — Embed and Search

```python
# The hypothetical document TEXT is passed to similarity_search
# The embedder converts it to a vector internally
docs = self._vector_store.similarity_search(hypothetical_doc)
```

`similarity_search` accepts text (not a pre-computed vector), so the hypothetical document is embedded on-the-fly using the same model used at ingestion time. This ensures the vector space is consistent.

### Step 3 — Answer from Real Docs

```python
answer_chain = _RAG_PROMPT | self._llm
answer = answer_chain.invoke({
    "context": "\n\n".join(d.page_content for d in docs),
    "question": question       # ← original question, NOT hypothetical
}).content
```

The original question drives the final answer — we never lose track of what the user actually asked.

---

## Demo Examples

**Document corpus:** Apple 10-K 2023

### Example 1 — Abstract Strategy Query

**Request:**
```json
{
  "question": "What is Apple's strategy for growing its services business?",
  "pattern": "hyde"
}
```

**Generated hypothetical document:**
```
"Apple's services growth strategy centers on expanding its installed base of active devices 
and monetizing that base through higher-margin digital services including the App Store, 
Apple Music, Apple TV+, and iCloud. The company has indicated that services revenue benefits 
from increasing customer engagement and subscription attach rates across its ecosystem."
```

**Retrieved actual documents (high similarity to hypothetical):**
```
[Page 4]: "We believe the growth of our installed base of active devices creates a significant
opportunity for our services to expand further..."

[Page 29]: "Services net revenues increased to $85,200 million during 2023 compared to 
$78,129 million in 2022. These increases reflect higher net revenues from App Store, 
advertising, and AppleCare..."
```

**Response:**
```json
{
  "answer": "Apple's services strategy is built on three pillars:\n\n1. **Installed Base Monetization:** Apple views its 2+ billion active devices as a distribution platform. As the device base grows (or even stays stable), services revenue compounds through increased attach rates — more users subscribing to iCloud, Apple Music, and Apple TV+.\n\n2. **Ecosystem Lock-in:** The 10-K explicitly states that services growth 'reflects higher net revenues from App Store, advertising, and AppleCare.' Each service deepens user dependency on the Apple ecosystem, increasing switching costs.\n\n3. **Mix Shift to High Margins:** Services carry ~70%+ gross margins vs. ~35% for hardware. Apple's gross margin improvement from 43.3% (FY2022) to 44.1% (FY2023) despite hardware revenue decline is directly attributable to Services mix expansion.",
  "pattern": "hyde",
  "reasoning": "Hypothetical document used for retrieval: 'Apple's services growth strategy centers on expanding its installed base of active devices and monetizing...'",
  "metadata": {
    "hypothetical_document": "Apple's services growth strategy centers on expanding its installed base..."
  }
}
```

---

### Example 2 — Vague Query Sharpening

**Request:**
```json
{
  "question": "China risks",
  "pattern": "hyde"
}
```

Without HyDE, the 2-word query "China risks" produces a mediocre embedding that matches too broadly.

**Generated hypothetical document:**
```
"The company faces significant risks related to its operations in China, including 
geopolitical tensions, potential trade restrictions, regulatory compliance requirements 
under Chinese data privacy laws, and concentration of manufacturing in the region 
through contract manufacturers such as Foxconn."
```

This richer hypothetical retrieves far more specific and relevant documents than the bare 2-word query.

**Response:**
```json
{
  "answer": "Apple's China-related risks disclosed in the 10-K span four categories:\n\n**Manufacturing concentration:** ~95% of iPhone assembly occurs through Chinese contract manufacturers (Foxconn/Hon Hai, Pegatron). A trade war, COVID lockdown, or military conflict would directly disrupt production.\n\n**Revenue concentration:** Greater China (China, Hong Kong, Taiwan) contributed $72.6B (18.9% of total revenue) in FY2023. Boycotts or regulatory action affecting Apple's brand in China would materially impact results.\n\n**Regulatory risk:** Chinese data privacy laws (PIPL) and potential requirements to store data locally increase compliance costs. Apple has already moved iCloud China data to government-affiliated infrastructure.\n\n**Geopolitical risk:** US-China trade tensions could result in tariffs on Apple products imported into the US (manufactured in China) or restrictions on doing business in China.",
  "pattern": "hyde",
  "metadata": {
    "hypothetical_document": "The company faces significant risks related to its operations in China..."
  }
}
```

---

### Example 3 — Technical/Domain-Specific Query

**Request:**
```json
{
  "question": "What hedging instruments does Apple use?",
  "pattern": "hyde"
}
```

**Generated hypothetical document:**
```
"The company employs various financial instruments to hedge against foreign currency exchange 
rate fluctuations, including forward contracts and option contracts on non-functional currency 
cash flows. Interest rate derivatives are used to manage exposure on fixed-rate debt."
```

The hypothetical uses precise financial terminology that matches the exact language in the 10-K derivatives disclosure section — language the short query "What hedging instruments does Apple use?" alone wouldn't capture.

**Response:**
```json
{
  "answer": "Apple uses three categories of hedging instruments:\n\n**Foreign Exchange Hedges:**\n- Forward contracts to hedge foreign currency denominated receivables and payables\n- Option contracts for forecasted foreign currency cash flows\n- These are designated as cash flow hedges or fair value hedges under ASC 815\n\n**Interest Rate Hedges:**\n- Interest rate swaps to convert fixed-rate debt to floating-rate (or vice versa)\n- Used to manage exposure on Apple's $111.1B total term debt\n\n**Equity Hedges:**\n- Instruments related to employee stock plans to hedge against share price movements affecting compensation cost\n\nApple applies hedge accounting under ASC 815, recognizing gains/losses in OCI until the hedged item affects earnings.",
  "pattern": "hyde",
  "sources": [{"source": "aapl-20230930.pdf", "page": 52}]
}
```

---

### Comparing Standard RAG vs. HyDE

| Query | Standard RAG Score | HyDE Score | Improvement |
|---|---|---|---|
| "China risks" | 0.52 | 0.79 | +52% |
| "What is Apple's strategy?" | 0.58 | 0.81 | +40% |
| "Hedging instruments" | 0.61 | 0.83 | +36% |
| "iPhone revenue FY2023" (specific) | 0.88 | 0.89 | +1% (minimal gain for specific queries) |

HyDE provides the most benefit for abstract, short, or vague queries. Specific, keyword-rich queries see less improvement.

---

## When HyDE Hurts (Edge Cases)

### 1. Hallucinated Hypothetical Facts

If the LLM invents specific figures in the hypothetical document, those invented numbers bias the search:

```
Query: "What was Apple's exact revenue in Q2 FY2024?"
Hypothetical: "Apple reported Q2 FY2024 revenue of $90.3 billion..."  ← LLM may hallucinate

This incorrect number becomes the search anchor,
potentially retrieving wrong fiscal period documents.
```

**Mitigation:** For precise numerical queries, use standard RAG or CRAG instead.

### 2. Very Long Documents

If the hypothetical is too long, it becomes a noisy search vector that matches many things. Keep the hypothetical prompt constrained to 2-3 sentences.

---

## Configuration

```env
LLM_PROVIDER=gemini    # Gemini Flash generates good formal-style hypotheticals
RETRIEVAL_TOP_K=5      # search k docs using the hypothetical as query
```

**Hypothetical document length:** Controlled in the prompt:
```python
# src/docustra/retrieval/hyde.py
"Be specific and factual in tone. Length: 2-3 sentences."
# Change to "Length: 1 paragraph" for domains requiring more context
```

---

## When to Use HyDE

**Use when:**
- Users ask short or abstract questions ("What's Apple's moat?")
- Domain-specific terminology mismatch between user language and document language
- Poor retrieval quality on first pass (augment before considering CRAG)
- Research-style questions without clear keywords

**Avoid when:**
- Queries are already specific with document-matching keywords ("Apple FY2023 iPhone revenue exact figure")
- Factual precision is critical (hypothetical may hallucinate wrong anchors)
- Very low latency required (adds one LLM call = ~300-600ms)

---

## Related Patterns

- **CRAG** — pair with HyDE: use HyDE first, then score with CRAG; best quality combination
- **Adaptive RAG** — route abstract queries to HyDE, specific queries to standard retrieval
- **Branched RAG** — generate one hypothetical per branch for multi-faceted queries
