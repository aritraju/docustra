"""
Agentic RAG
───────────
The LLM acts as an orchestrator in a LangGraph ReAct loop.
Available tools: vector_search, web_search, calculator.
The agent iterates until it has sufficient context to answer.
"""
from typing import Annotated, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from docustra.core import get_logger, get_settings
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _make_vector_search_tool(vector_store: VectorStore):
    @tool
    def vector_search(query: str) -> str:
        """Search the indexed enterprise documents for relevant context."""
        docs = vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found."
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

    return vector_search


class AgenticRAG(BaseRAGStrategy):
    pattern = RAGPattern.AGENTIC

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        vector_store = VectorStore(get_embeddings())

        tools = [_make_vector_search_tool(vector_store)]
        if settings.tavily_api_key and settings.tavily_api_key != "your_tavily_api_key_here":
            import os
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
            tools.append(TavilySearchResults(max_results=3))

        llm_with_tools = self._llm.bind_tools(tools)

        def agent_node(state: AgentState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")
        self._graph = graph.compile()

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Agentic RAG query", question=question[:80])
        system_msg = HumanMessage(
            content=(
                "You are a document intelligence assistant. Use the available tools to search "
                "documents and answer questions thoroughly. Always cite your sources.\n\n"
                f"Question: {question}"
            )
        )
        result = self._graph.invoke({"messages": [system_msg]})
        messages = result["messages"]

        final_answer = ""
        tool_calls_made = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_made.extend([tc["name"] for tc in msg.tool_calls])
                else:
                    final_answer = msg.content

        return RAGResponse(
            answer=final_answer or "No answer generated.",
            pattern=self.pattern,
            reasoning=f"Agent used {len(tool_calls_made)} tool calls: {tool_calls_made}",
            metadata={"tool_calls": tool_calls_made, "total_messages": len(messages)},
        )
