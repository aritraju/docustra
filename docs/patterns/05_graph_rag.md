# Graph RAG

## Overview

Graph RAG is one of the most significant recent advances in RAG architecture. It builds a **knowledge graph** on top of documents — mapping entities and their relationships — and uses this graph as an additional retrieval source alongside traditional vector search.

The fundamental insight is: **some questions are not about finding relevant passages, they're about connecting dots across passages**. Vector search excels at finding semantically similar text; graph search excels at multi-hop relationship traversal.

---

## The Problem It Solves

Consider this question:
> "How does the EU Digital Markets Act regulation affect Apple's App Store revenue and what vendors might benefit?"

A vector search will find documents mentioning the DMA, and others mentioning App Store revenue. But it won't automatically connect:
- **DMA** → *requires* → **third-party app stores**
- **App Store** → *generates* → **$85B+ services revenue**
- **Spotify**, **Epic Games** → *are vendors* → *affected by* → **App Store rules**
- **Spotify** → *could benefit* → *from* → **alternative distribution**

Graph RAG answers this by traversing the knowledge graph to surface these multi-hop relationships.

---

## Architecture

### Ingestion Phase (Build the Graph)

```
Document Text
      │
      ▼
┌─────────────────────────────────────┐
│         EntityExtractor (LLM)        │
│                                     │
│  Input: "Apple faces EU DMA rules.  │
│   Spotify could benefit..."         │
│                                     │
│  Output:                            │
│  entities: [                        │
│    {name: "Apple", type: COMPANY},  │
│    {name: "EU DMA", type: REGULATION}│
│    {name: "Spotify", type: COMPANY} │
│  ]                                  │
│  relationships: [                   │
│    {from: "EU DMA", type: REGULATES,│
│     to: "Apple"},                   │
│    {from: "Apple", type: OPERATES,  │
│     to: "App Store"}                │
│  ]                                  │
└──────────────┬──────────────────────┘
               │
               ▼
         Neo4j Graph
    ┌──────────────────────────┐
    │  (Apple)─[REGULATES]─▶  │
    │  (EU DMA)               │
    │       │                 │
    │  [OPERATES]             │
    │       │                 │
    │       ▼                 │
    │  (App Store)            │
    │       │                 │
    │  [COMPETES_WITH]        │
    │       │                 │
    │       ▼                 │
    │  (Spotify)              │
    └──────────────────────────┘
```

### Query Phase

```
          User Question
                │
    ┌───────────┴───────────┐
    │ Run in parallel:       │
    │                        │
    ▼                        ▼
VectorStore              GraphStore
.similarity_search()    .get_entity_context()
    │                    │
    │  LLM first         │  Cypher traversal:
    │  extracts          │  MATCH (e)-[r*1..2]-(n)
    │  entities          │  WHERE e.name IN entities
    │  from question     │
    │                        │
    └─────────┬──────────────┘
              │
              ▼
    ┌───────────────────────────────────────┐
    │              LLM Generator             │
    │                                       │
    │  Context 1: Retrieved text passages   │
    │  Context 2: Knowledge graph triples   │
    │  → Combined answer with relationships │
    └───────────────────────────────────────┘
```

---

## Implementation Walkthrough

**Files:** `src/docustra/retrieval/graph.py`, `src/docustra/graph/`

### Entity Extraction at Ingestion

```python
# src/docustra/graph/extractor.py
_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract entities and relationships from the text.
Return valid JSON with this exact structure:
{
  "entities": [{"name": "...", "type": "COMPANY|PERSON|REGULATION|PRODUCT|LOCATION|CONCEPT"}],
  "relationships": [{"from": "...", "type": "...", "to": "..."}]
}"""),
    ("human", "{text}"),
])
```

**Sample extraction from Apple 10-K:**

Input text:
```
"Apple faces regulatory scrutiny from the European Commission under the Digital Markets Act,
which requires Apple to allow alternative app stores on iOS devices. Spotify and Epic Games
have been vocal critics of Apple's App Store commission structure."
```

Extracted graph:
```json
{
  "entities": [
    {"name": "Apple", "type": "COMPANY"},
    {"name": "European Commission", "type": "COMPANY"},
    {"name": "Digital Markets Act", "type": "REGULATION"},
    {"name": "iOS", "type": "PRODUCT"},
    {"name": "App Store", "type": "PRODUCT"},
    {"name": "Spotify", "type": "COMPANY"},
    {"name": "Epic Games", "type": "COMPANY"}
  ],
  "relationships": [
    {"from": "European Commission", "type": "ENFORCES", "to": "Digital Markets Act"},
    {"from": "Digital Markets Act", "type": "REGULATES", "to": "Apple"},
    {"from": "Digital Markets Act", "type": "REQUIRES", "to": "App Store"},
    {"from": "App Store", "type": "RUNS_ON", "to": "iOS"},
    {"from": "Spotify", "type": "CRITICIZES", "to": "App Store"},
    {"from": "Epic Games", "type": "CRITICIZES", "to": "App Store"}
  ]
}
```

### Neo4j Storage

```python
# Entities stored as labeled nodes
MERGE (e:COMPANY {name: "Apple"})
MERGE (e:REGULATION {name: "Digital Markets Act"})

# Relationships stored as typed edges
MATCH (a {name: "Digital Markets Act"}), (b {name: "Apple"})
MERGE (a)-[:REGULATES]->(b)
```

### Query-time Graph Traversal

```python
# src/docustra/storage/graph_store.py
def get_entity_context(self, entity_names: list[str]) -> str:
    cypher = """
    MATCH (e) WHERE e.name IN $names
    OPTIONAL MATCH (e)-[r]-(neighbor)
    RETURN e.name AS entity, labels(e) AS type,
           collect(DISTINCT {rel: type(r), neighbor: neighbor.name}) AS connections
    """
    rows = self.run_query(cypher, {"names": entity_names})
    # Formats as: "Apple (COMPANY): REGULATES <- Digital Markets Act; OPERATES -> App Store"
```

### Combined Retrieval

```python
# src/docustra/retrieval/graph.py
def query(self, question: str) -> RAGResponse:
    entities = self._extract_entities(question)
    
    # Parallel execution
    docs = self._vector_store.similarity_search(question)
    graph_context = self._graph_store.get_entity_context(entities)
    
    answer = (_RAG_PROMPT | self._llm).invoke({
        "question": question,
        "doc_context": "\n\n".join(d.page_content for d in docs),
        "graph_context": graph_context
    }).content
```

---

## Demo Examples

**Document corpus:** Apple 10-K 2023 + knowledge graph built from same

### Example 1 — Multi-hop Entity Relationship

**Request:**
```json
{
  "question": "How does the EU Digital Markets Act affect Apple's relationship with Spotify and Epic Games?",
  "pattern": "graph"
}
```

**Entity extraction from question:**
```
Entities found: ["EU Digital Markets Act", "Apple", "Spotify", "Epic Games"]
```

**Graph traversal result:**
```
EU Digital Markets Act (REGULATION):
  REGULATES → Apple
  REQUIRES → App Store alternative distribution

Apple (COMPANY):
  REGULATED_BY ← EU Digital Markets Act
  OPERATES → App Store
  COMPETES_WITH → Spotify (music streaming)

App Store (PRODUCT):
  OPERATED_BY ← Apple
  CRITICIZED_BY ← Spotify
  CRITICIZED_BY ← Epic Games
  SUBJECT_TO → 30% commission structure

Spotify (COMPANY):
  CRITICIZES → App Store
  BENEFITS_FROM → DMA alternative distribution rules

Epic Games (COMPANY):
  CRITICIZES → App Store
  FILED_LAWSUIT_AGAINST → Apple
```

**Response:**
```json
{
  "answer": "The EU Digital Markets Act creates a cascading effect on Apple's relationships with Spotify and Epic Games:\n\n**Direct Impact on Apple:** The DMA designates Apple as a 'gatekeeper,' requiring it to allow alternative app marketplaces on iOS in the EU. This directly undermines the App Store's monopoly on iOS software distribution.\n\n**Spotify's Position:** Spotify has been one of the most vocal critics of Apple's App Store, objecting specifically to the 30% commission on in-app purchases and the prohibition on linking to external payment methods. The DMA's requirements benefit Spotify by:\n- Allowing direct browser-based payments without Apple's commission\n- Permitting alternative app store distribution\n- Potentially reducing Spotify's effective cost of serving EU customers\n\n**Epic Games' Position:** Epic Games filed an antitrust lawsuit against Apple in 2020 over App Store rules, which aligns directly with DMA provisions. The DMA effectively validates Epic's legal arguments in the EU jurisdiction.\n\n**Financial Impact on Apple:** The 10-K discloses that App Store-related revenue is part of the $85.2B Services segment. While Apple has implemented DMA compliance (e.g., Core Technology Fee structure), analysts estimate EU DMA compliance could reduce Services revenue by $1-3B annually.",
  "pattern": "graph",
  "reasoning": "Entities extracted: ['EU Digital Markets Act', 'Apple', 'Spotify', 'Epic Games']. Graph context lines: 14.",
  "metadata": {
    "entities": ["EU Digital Markets Act", "Apple", "Spotify", "Epic Games"],
    "graph_context_found": true
  }
}
```

---

### Example 2 — Supply Chain Relationship Traversal

**Request:**
```json
{
  "question": "What is the relationship between Apple, TSMC, and Taiwan in the context of supply chain risk?",
  "pattern": "graph"
}
```

**Graph traversal:**
```
Apple (COMPANY):
  MANUFACTURES_WITH → TSMC
  MANUFACTURES_WITH → Foxconn
  SUPPLIES_MARKET → Taiwan (Greater China region)

TSMC (COMPANY):
  MANUFACTURES_FOR → Apple Silicon (A17 Pro, M3)
  LOCATED_IN → Taiwan
  SOLE_SUPPLIER_OF → Advanced 3nm chips for Apple

Taiwan (LOCATION):
  HOSTS → TSMC headquarters
  SUBJECT_TO → Geopolitical tension (US-China)
  COVERED_BY → US CHIPS Act investment
```

**Response:**
```json
{
  "answer": "The Apple-TSMC-Taiwan triangle represents Apple's most concentrated supply chain risk:\n\n**Apple → TSMC:** TSMC manufactures all of Apple's custom silicon (A-series, M-series, S-series chips) using cutting-edge process nodes (3nm and below). Apple is TSMC's largest customer (~25% of revenue). This relationship is effectively a single point of failure — no other foundry can manufacture Apple Silicon at this scale or process maturity.\n\n**TSMC → Taiwan:** TSMC's primary manufacturing facilities are in Taiwan. While TSMC is building fabs in Arizona (2nm planned for 2026), these represent a fraction of total capacity.\n\n**Geopolitical Risk:** The 10-K explicitly cites 'international geopolitical tensions' and Taiwan as a risk factor. Cross-strait tensions between China and Taiwan create a scenario where Apple's entire chip supply could be disrupted. The US CHIPS Act partially mitigates this by incentivizing TSMC US expansion, but full diversification is a decade-long effort.\n\n**Compounding factor:** Apple's Greater China revenue ($72.6B, 19% of total) means a Taiwan conflict would simultaneously disrupt both supply (chips) and demand (China sales).",
  "pattern": "graph",
  "metadata": {
    "entities": ["Apple", "TSMC", "Taiwan"],
    "graph_context_found": true
  }
}
```

---

### Example 3 — When Graph Context Is Empty

**Request:**
```json
{
  "question": "What font does Apple use in its annual report?",
  "pattern": "graph"
}
```

**Entity extraction:** `[]` — no named entities found.

**Behavior:** Falls back gracefully to vector-only retrieval.

**Response:**
```json
{
  "answer": "The 10-K filing does not specify the typography used in its production. SEC filings follow EDGAR formatting standards rather than design guidelines.",
  "pattern": "graph",
  "reasoning": "Entities extracted: []. Graph context found: False.",
  "metadata": {
    "entities": [],
    "graph_context_found": false
  }
}
```

---

## Neo4j Browser Exploration

After ingestion, explore the knowledge graph visually at `http://localhost:7474`:

```cypher
-- See all entity types
MATCH (n) RETURN DISTINCT labels(n) AS type, count(n) AS count

-- See Apple's complete neighborhood
MATCH (apple:COMPANY {name: "Apple"})-[r]-(neighbor)
RETURN apple, r, neighbor LIMIT 50

-- Find path between two entities
MATCH p = shortestPath(
  (dma:REGULATION {name: "Digital Markets Act"})-[*..5]-(spotify:COMPANY {name: "Spotify"})
)
RETURN p

-- Most connected entities
MATCH (n)-[r]-()
RETURN n.name, count(r) AS connections
ORDER BY connections DESC LIMIT 10
```

---

## Configuration

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=docustra_local
```

**Graph traversal depth** (edit `graph_store.py`):
```python
def find_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
    # depth=1: direct connections only
    # depth=2: 2-hop relationships (default)
    # depth=3: 3-hop (slower, more comprehensive)
```

---

## When to Use Graph RAG

**Use when:**
- Questions ask "how does X affect Y" or "what is the relationship between X and Z"
- Your corpus has rich entity density (regulations, companies, products, people)
- Multi-hop reasoning: A → B → C connections
- Compliance questions spanning multiple regulatory frameworks

**Avoid when:**
- Documents are conceptual/abstract with few named entities
- Building the KG is cost-prohibitive (each chunk requires an LLM extraction call)
- Questions are factual lookups from a single passage (vector search is sufficient)

---

## Graph Quality Matters

The quality of Graph RAG depends entirely on the quality of entity extraction:

| Extraction quality | Graph RAG quality |
|---|---|
| Poor (generic entities) | No better than vector search |
| Good (domain-specific) | 30-50% better on relationship questions |
| Excellent (validated) | Transformative for complex reasoning |

**Improving extraction:** Fine-tune the extraction prompt with domain-specific entity types. For financial documents, add: `FINANCIAL_METRIC`, `FISCAL_PERIOD`, `REGULATORY_BODY`, `ACCOUNTING_STANDARD`.

---

## Related Patterns

- **Agentic RAG** — the agent can query the graph as one of its tools dynamically
- **Branched RAG** — branch on entities: one branch per key entity, synthesize relationships
- **CRAG** — graph context can supplement low-scoring vector results as a correction source
