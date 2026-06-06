# Agentic RAG

## Overview

Agentic RAG is widely considered **the direction the entire RAG field is moving**. Rather than a fixed pipeline (retrieve → generate), it hands control to an LLM that decides at each step what action to take next: search for more information, call an external API, perform a calculation, or determine it has enough context to answer.

The LLM operates in a **ReAct loop** (Reason + Act) until it produces a final answer, making it transformative for queries that require exploration, validation, or multi-tool reasoning.

---

## The Problem It Solves

Standard RAG retrieves once and answers once. If the retrieved documents are incomplete, the system either hallucinates or gives a partial answer. There's no mechanism for the system to recognize its own knowledge gaps and fill them.

Agentic RAG solves this by letting the LLM:
1. Recognize when it needs more information
2. Decide *which* tool to use (vector search, web search, calculator)
3. Evaluate whether the retrieved information is sufficient
4. Iterate until the answer is complete

---

## Architecture

```
User Question
      │
      ▼
┌──────────────────────────────────────────────────────┐
│                  LangGraph StateGraph                 │
│                                                       │
│   ┌─────────────────────────────────────────┐        │
│   │              AgentState                  │        │
│   │   messages: [HumanMsg, AIMsg, ToolMsg]   │        │
│   └──────────────────┬──────────────────────┘        │
│                       │                               │
│              ┌────────▼────────┐                      │
│   ┌──────────│  agent_node     │──────────┐           │
│   │          │  LLM + tools    │          │           │
│   │          └────────┬────────┘          │           │
│   │                   │                   │           │
│   │      has tool_calls?                  │           │
│   │       YES │         NO (END)          │           │
│   │           ▼                           │           │
│   │  ┌────────────────┐                  │           │
│   │  │  tools_node     │                  │           │
│   │  │  ToolNode       │                  │           │
│   │  │  ┌────────────┐ │                  │           │
│   └──│  │vector_search│ │──────────────────┘          │
│      │  ├────────────┤ │                              │
│      │  │ web_search  │ │  (loop back to agent_node)  │
│      │  └────────────┘ │                              │
│      └────────────────┘                               │
└──────────────────────────────────────────────────────┘
      │
      ▼
 Final AIMessage (no tool_calls)
      │
      ▼
   RAGResponse
```

---

## Implementation Walkthrough

**File:** `src/docustra/retrieval/agentic.py`

### State Definition

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` is a LangGraph reducer that appends new messages to the list rather than replacing them. This gives the agent full conversation history.

### Tool Registration

```python
@tool
def vector_search(query: str) -> str:
    """Search the indexed enterprise documents for relevant context."""
    docs = vector_store.similarity_search(query, k=4)
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source')}, Page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )
```

The docstring is critical — the LLM reads it to decide when to invoke this tool.

### Graph Construction

```python
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"   # continue loop
    return END           # agent is done

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")   # always loop back
```

### Loop Execution

```python
def query(self, question: str) -> RAGResponse:
    result = self._graph.invoke({
        "messages": [HumanMessage(content=f"Question: {question}")]
    })
    # Collect tool calls made and final answer
    tool_calls_made = [tc["name"] for msg in result["messages"]
                       if isinstance(msg, AIMessage) and msg.tool_calls
                       for tc in msg.tool_calls]
    final_answer = next(
        msg.content for msg in reversed(result["messages"])
        if isinstance(msg, AIMessage) and not msg.tool_calls
    )
```

---

## Demo Example

**Document corpus:** Apple 10-K 2023 + SEC regulatory guidance docs

### Single-hop Query (Agent uses 1 tool call)

**Request:**
```json
{
  "question": "What is Apple's current ratio for fiscal year 2023?",
  "pattern": "agentic"
}
```

**Agent trace:**
```
[Turn 1] Agent → vector_search("Apple current ratio fiscal year 2023 balance sheet")
[Tool]   → "Current assets: $143,566M | Current liabilities: $145,308M..."
[Turn 2] Agent → Final answer (no more tool calls needed)
```

**Response:**
```json
{
  "answer": "Apple's current ratio for fiscal year 2023 was approximately 0.988 (current assets of $143,566M divided by current liabilities of $145,308M). This ratio below 1.0 is typical for Apple due to its large accounts payable from manufacturing partners and deferred revenue from services, offset by significant cash reserves held as long-term investments. Source: Apple 10-K 2023, Balance Sheet, Page 48.",
  "pattern": "agentic",
  "reasoning": "Agent used 1 tool calls: ['vector_search']",
  "metadata": {
    "tool_calls": ["vector_search"],
    "total_messages": 3
  }
}
```

---

### Multi-hop Query (Agent iterates 3 times)

**Request:**
```json
{
  "question": "How has Apple's R&D spending as a percentage of revenue changed over the last 3 years, and how does this compare to the industry average cited in the filing?",
  "pattern": "agentic"
}
```

**Agent trace:**
```
[Turn 1] Agent → vector_search("Apple R&D research development expense 2021 2022 2023")
[Tool]   → "R&D: $26,251M (2023), $26,251M (2022), $21,914M (2021)..."

[Turn 2] Agent → vector_search("Apple total net revenue 2021 2022 2023")
[Tool]   → "Net revenue: $383,285M (2023), $394,328M (2022), $365,817M (2021)..."

[Turn 3] Agent → vector_search("technology industry R&D spending average benchmark comparison")
[Tool]   → "No specific industry average cited in filing. Apple does not compare..."

[Turn 4] Agent → Final answer (sufficient context gathered)
```

**Response:**
```json
{
  "answer": "Apple's R&D spending as a percentage of revenue has been: FY2021: 6.0% ($21.9B / $365.8B), FY2022: 6.7% ($26.3B / $394.3B), FY2023: 6.8% ($26.3B / $383.3B). This shows a modest but consistent increase in R&D intensity. Notably, the 10-K filing does not cite an explicit industry average for comparison — Apple's disclosures focus on absolute R&D figures rather than benchmarking against peers. For context, industry peers like Microsoft spend ~13% and Alphabet ~15% on R&D, though Apple's hardware-heavy model makes direct comparison difficult.",
  "pattern": "agentic",
  "reasoning": "Agent used 3 tool calls: ['vector_search', 'vector_search', 'vector_search']",
  "metadata": {
    "tool_calls": ["vector_search", "vector_search", "vector_search"],
    "total_messages": 7
  }
}
```

---

### Query Requiring Web Search Fallback

When Tavily API key is configured and the document corpus lacks coverage:

**Request:**
```json
{
  "question": "What is Apple's stock price today and how does it compare to the book value per share in the 2023 10-K?",
  "pattern": "agentic"
}
```

**Agent trace:**
```
[Turn 1] Agent → vector_search("Apple book value per share stockholders equity 2023")
[Tool]   → "Total stockholders equity: -$13,306M | Shares outstanding: 15,634M..."

[Turn 2] Agent → tavily_search("Apple AAPL stock price today 2024")
[Tool]   → "AAPL: $189.30 (as of market close)..."

[Turn 3] Agent → Final answer
```

**Response:**
```json
{
  "answer": "From the 2023 10-K: Apple's total stockholders' equity was -$13,306M with 15,634M shares outstanding, giving a book value per share of approximately -$0.85 (negative, because Apple returns more capital than it retains). The current market price of approximately $189.30 represents a price-to-book ratio that is technically undefined (negative book value), which is common for capital-light, high-return companies. Investors price Apple on earnings and cash flow multiples rather than book value.",
  "reasoning": "Agent used 2 tool calls: ['vector_search', 'tavily_search']",
  "metadata": {
    "tool_calls": ["vector_search", "tavily_search"],
    "total_messages": 5
  }
}
```

---

## Observability in Arize Phoenix

Every agent iteration is a separate trace span. In Phoenix (`http://localhost:6006`), you can see:

```
Query: "How has Apple's R&D..."
├── agent_node (Turn 1)
│   ├── LLM call: 847 tokens
│   └── tool_call: vector_search
├── tools_node: vector_search execution
├── agent_node (Turn 2)
│   ├── LLM call: 1,203 tokens
│   └── tool_call: vector_search
├── tools_node: vector_search execution
├── agent_node (Turn 3)
│   ├── LLM call: 1,892 tokens
│   └── tool_call: vector_search
├── tools_node: vector_search execution
└── agent_node (Turn 4)
    └── LLM call: 2,341 tokens → FINAL ANSWER
```

---

## Configuration

```env
TAVILY_API_KEY=your_key   # enables web search tool (optional)
RETRIEVAL_TOP_K=4         # docs returned per vector_search call
```

**Adding custom tools:** Extend the `tools` list in `AgenticRAG.__init__()`:

```python
from langchain_core.tools import tool

@tool
def financial_calculator(expression: str) -> str:
    """Evaluate a financial arithmetic expression."""
    return str(eval(expression))  # use safe_eval in production

tools = [vector_search_tool, financial_calculator]
```

---

## When to Use Agentic RAG

**Use when:**
- Queries require discovering what to search for (open-ended exploration)
- The answer requires combining information from multiple searches
- You need to invoke external tools (APIs, databases, calculators)
- Query complexity is unpredictable

**Avoid when:**
- Latency < 2 seconds is required (each loop adds ~800ms-1.5s)
- The query pattern is known and fixed (use a simpler targeted pattern)
- The LLM might loop indefinitely (always set `recursion_limit` in production)

---

## Production Considerations

```python
# Set recursion limit to prevent infinite loops
graph.compile(checkpointer=None)
config = {"recursion_limit": 10}
result = self._graph.invoke(state, config=config)
```

Add a **stopping condition** beyond "no tool calls":
- Max iterations reached
- Answer confidence threshold met
- Specific completion token in response

---

## Performance Characteristics

| Scenario | Agent Iterations | Typical Latency | Token Cost |
|---|---|---|---|
| Simple lookup | 1 | ~1.5s | Low |
| Multi-hop research | 3-4 | ~4-6s | Medium |
| Cross-domain with web | 2-3 | ~5-8s | Medium |
| Open-ended exploration | 5-8 | ~10-15s | High |

---

## Related Patterns

- **Branched RAG** — predetermined parallel branches vs. dynamic agent decisions
- **Adaptive RAG** — routes to Agentic for complex queries; less flexible but more predictable
- **CRAG** — simpler fallback mechanism; Agentic RAG subsumes CRAG's capability
