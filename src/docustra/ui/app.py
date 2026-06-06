import requests
import streamlit as st

API_BASE = "http://localhost:8000"

PATTERN_DESCRIPTIONS = {
    "adaptive": "Routes query to the right retrieval depth — trivial, simple, or complex.",
    "agentic": "LangGraph agent loop with tool use — searches until confident.",
    "branched": "Decomposes query → parallel sub-retrievals → synthesis.",
    "corrective": "Scores retrieved docs; falls back to web search if confidence is low.",
    "graph": "Augments vector search with Neo4j knowledge graph entity relationships.",
    "hyde": "Generates a hypothetical document first, uses it as the search vector.",
    "multimodal": "Retrieves text + describes images/charts from documents.",
    "self_rag": "LLM self-critiques with [Retrieve], [Relevant], [Supported] tokens.",
}


def main():
    st.set_page_config(
        page_title="Docustra",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Docustra")
    st.caption("Enterprise Document Intelligence — Advanced RAG Pattern Showcase")

    with st.sidebar:
        st.header("⚙️ Configuration")
        pattern = st.selectbox(
            "RAG Pattern",
            options=list(PATTERN_DESCRIPTIONS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        st.info(PATTERN_DESCRIPTIONS[pattern])

        st.divider()
        st.subheader("📄 Ingest Document")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        build_graph = st.checkbox("Build Knowledge Graph", value=True)
        if uploaded and st.button("Ingest"):
            with st.spinner("Ingesting..."):
                r = requests.post(
                    f"{API_BASE}/ingest/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    data={"build_graph": str(build_graph).lower()},
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"✅ {data['chunks_indexed']} chunks indexed, "
                        f"{data['graph_entities']} entities in graph"
                    )
                else:
                    st.error(f"Ingestion failed: {r.text}")

        st.divider()
        if st.button("🔍 Check Services"):
            r = requests.get(f"{API_BASE}/health")
            if r.ok:
                data = r.json()
                for svc, status in data["services"].items():
                    icon = "✅" if status == "ok" else "⚠️"
                    st.write(f"{icon} {svc}: {status}")

    # Main query interface
    question = st.text_area(
        "Ask a question",
        placeholder="e.g. What are the main risk factors disclosed?",
        height=100,
    )

    if st.button("🔎 Query", type="primary", disabled=not question.strip()):
        with st.spinner(f"Running {pattern.replace('_', ' ').title()} RAG..."):
            r = requests.post(
                f"{API_BASE}/query",
                json={"question": question, "pattern": pattern},
            )

        if r.status_code == 200:
            data = r.json()

            st.subheader("Answer")
            st.markdown(data["answer"])

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("🧠 Reasoning & Reflection"):
                    st.text(data["reasoning"] or "No reasoning logged.")
                with st.expander("🔧 Metadata"):
                    st.json(data["metadata"])
            with col2:
                with st.expander(f"📎 Sources ({len(data['sources'])})"):
                    for s in data["sources"]:
                        st.markdown(f"**{s.get('source', 'unknown')}** — Page {s.get('page', '?')}")
                        st.caption(s.get("content", "")[:200])
                        st.divider()
        else:
            st.error(f"Query failed ({r.status_code}): {r.text}")


if __name__ == "__main__":
    main()
