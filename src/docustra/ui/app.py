"""
Docustra — Streamlit UI  (2026 redesign)
Dark-mode-first, indigo/cyan gradient palette, card-based glassmorphism layout.

Three tabs:
  📄 Document Intelligence  — ingest + chunking strategy + live params
  🔍 RAG Query              — 8 patterns with icon badges
  📊 System Dashboard       — health, stats, strategy reference
"""

from __future__ import annotations

import json
import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Design system — full CSS injection
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
:root {
    --bg:           #0A0F1E;
    --surface:      #111827;
    --surface-alt:  #1C2436;
    --border:       #2D3748;
    --border-glow:  rgba(99,102,241,0.4);
    --primary:      #6366F1;
    --primary-soft: #818CF8;
    --cyan:         #06B6D4;
    --emerald:      #10B981;
    --amber:        #F59E0B;
    --red:          #EF4444;
    --text:         #F1F5F9;
    --text-muted:   #94A3B8;
    --grad:         linear-gradient(135deg,#6366F1 0%,#06B6D4 100%);
    --grad-soft:    linear-gradient(135deg,rgba(99,102,241,.15) 0%,rgba(6,182,212,.08) 100%);
}
.stApp { background:var(--bg) !important; color:var(--text) !important; }
.block-container { padding:1.5rem 2rem 3rem !important; max-width:1200px !important; }
section[data-testid="stSidebar"] { display:none; }
#MainMenu,footer,header { visibility:hidden; }

/* ── Hero ─────────────────────────────────────── */
.hero {
    background:var(--grad);
    border-radius:16px;
    padding:28px 36px;
    margin-bottom:24px;
    position:relative;
    overflow:hidden;
}
.hero::before {
    content:"";
    position:absolute;
    top:-60px;right:-60px;
    width:220px;height:220px;
    border-radius:50%;
    background:rgba(255,255,255,.06);
}
.hero h1 { font-size:2rem;font-weight:800;color:#fff !important;margin:0 0 4px;letter-spacing:-.5px; }
.hero p  { color:rgba(255,255,255,.82);font-size:.95rem;margin:0; }

/* ── Tabs ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:var(--surface) !important;
    border-radius:12px !important;
    padding:5px !important;
    gap:4px !important;
    border:1px solid var(--border) !important;
    margin-bottom:24px !important;
}
.stTabs [data-baseweb="tab"] {
    background:transparent !important;
    border-radius:8px !important;
    color:var(--text-muted) !important;
    font-weight:500 !important;
    font-size:.9rem !important;
    padding:10px 22px !important;
    transition:all .2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color:var(--text) !important;
    background:var(--surface-alt) !important;
}
.stTabs [aria-selected="true"] {
    background:var(--grad) !important;
    color:#fff !important;
    box-shadow:0 2px 12px rgba(99,102,241,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:0 !important; }

/* ── Metrics ───────────────────────────────────── */
[data-testid="stMetric"] {
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    border-radius:10px !important;
    padding:16px 20px !important;
}
[data-testid="stMetricLabel"] { color:var(--text-muted) !important;font-size:.8rem !important; }
[data-testid="stMetricValue"] { color:var(--text) !important;font-size:1.6rem !important;font-weight:700 !important; }

/* ── Badges ────────────────────────────────────── */
.badge {
    display:inline-block;
    padding:3px 10px;
    border-radius:20px;
    font-size:.72rem;
    font-weight:600;
    letter-spacing:.04em;
    text-transform:uppercase;
}
.bi { background:rgba(99,102,241,.18);color:#818CF8;border:1px solid rgba(99,102,241,.3); }
.bc { background:rgba(6,182,212,.18);color:#22D3EE;border:1px solid rgba(6,182,212,.3); }
.bg { background:rgba(16,185,129,.18);color:#34D399;border:1px solid rgba(16,185,129,.3); }
.ba { background:rgba(245,158,11,.18);color:#FCD34D;border:1px solid rgba(245,158,11,.3); }
.br { background:rgba(239,68,68,.18);color:#FCA5A5;border:1px solid rgba(239,68,68,.3); }
.bm { background:rgba(148,163,184,.12);color:var(--text-muted);border:1px solid var(--border); }

/* ── Pattern cards ─────────────────────────────── */
.pcard {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:10px;
    padding:14px 16px;
    transition:all .18s ease;
    height:100%;
    min-height:130px;
}
.pcard:hover {
    border-color:var(--primary);
    background:var(--surface-alt);
    transform:translateY(-1px);
    box-shadow:0 4px 16px rgba(99,102,241,.2);
}
.pcard.sel {
    border-color:var(--primary);
    background:var(--grad-soft);
    box-shadow:0 0 0 1px var(--primary),0 4px 16px rgba(99,102,241,.25);
}
.pcard .pi { font-size:1.4rem;margin-bottom:4px; }
.pcard .pn { font-weight:700;font-size:.88rem;color:var(--text); }
.pcard .pd { font-size:.76rem;color:var(--text-muted);line-height:1.4;margin-top:5px; }

/* ── Cards ─────────────────────────────────────── */
.card {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:12px;
    padding:18px 22px;
    margin-bottom:14px;
    transition:border-color .2s;
}
.card:hover { border-color:var(--border-glow); }
.ct { font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:10px; }

/* ── Svc cards ─────────────────────────────────── */
.svc {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:10px;
    padding:16px 20px;
}

/* ── Inputs ────────────────────────────────────── */
.stTextArea textarea,
.stTextInput input,
.stNumberInput input {
    background:var(--surface-alt) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    color:var(--text) !important;
    font-size:.9rem !important;
}
.stTextArea textarea:focus,
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color:var(--primary) !important;
    box-shadow:0 0 0 2px rgba(99,102,241,.25) !important;
}
.stSelectbox>div>div {
    background:var(--surface-alt) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    color:var(--text) !important;
}
label { color:var(--text-muted) !important;font-size:.82rem !important;font-weight:500 !important; }

/* ── Buttons ───────────────────────────────────── */
.stButton>button[kind="primary"] {
    background:var(--grad) !important;
    border:none !important;
    border-radius:8px !important;
    color:#fff !important;
    font-weight:600 !important;
    font-size:.9rem !important;
    padding:10px 24px !important;
    transition:all .2s ease !important;
    box-shadow:0 2px 12px rgba(99,102,241,.35) !important;
}
.stButton>button[kind="primary"]:hover {
    box-shadow:0 4px 20px rgba(99,102,241,.55) !important;
    transform:translateY(-1px) !important;
}
.stButton>button[kind="primary"]:disabled {
    opacity:.4 !important;
    transform:none !important;
    box-shadow:none !important;
}
.stButton>button:not([kind="primary"]) {
    background:var(--surface-alt) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    color:var(--text) !important;
    font-weight:500 !important;
    font-size:.83rem !important;
}
.stButton>button:not([kind="primary"]):hover {
    border-color:var(--primary-soft) !important;
    color:var(--primary-soft) !important;
}

/* ── File uploader ─────────────────────────────── */
[data-testid="stFileUploader"] {
    border:2px dashed var(--border) !important;
    border-radius:12px !important;
    background:var(--surface) !important;
    transition:border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color:var(--primary) !important; }

/* ── Alerts ─────────────────────────────────────── */
.stAlert { border-radius:8px !important;font-size:.87rem !important; }

/* ── Toggle ─────────────────────────────────────── */
[data-testid="stToggle"] span[aria-checked="true"] { background:var(--primary) !important; }

/* ── Progress ───────────────────────────────────── */
[data-testid="stProgressBar"]>div>div { background:var(--grad) !important;border-radius:4px !important; }
[data-testid="stProgressBar"]>div { background:var(--surface-alt) !important;border-radius:4px !important; }

/* ── Expanders ──────────────────────────────────── */
details {
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
}
details summary { color:var(--text) !important;font-weight:500 !important;font-size:.88rem !important;padding:8px 16px !important; }

/* ── Misc ───────────────────────────────────────── */
hr { border-color:var(--border) !important;margin:20px 0 !important; }
pre,code { background:var(--surface-alt) !important;color:#A5B4FC !important;border-radius:6px !important;font-size:.82rem !important; }
::-webkit-scrollbar { width:5px;height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--primary); }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RAG pattern registry
# ─────────────────────────────────────────────────────────────────────────────

_GH = "https://github.com/aritraju/docustra/blob/main"

RAG_PATTERNS: dict[str, dict] = {
    "adaptive": {
        "icon": "🔀",
        "label": "Adaptive",
        "badge": "General Purpose",
        "bc": "bi",
        "desc": "Routes by complexity: trivial→direct, simple→1× search, complex→multi-step.",
        "doc": f"{_GH}/docs/patterns/01_adaptive_rag.md",
    },
    "agentic": {
        "icon": "🤖",
        "label": "Agentic",
        "badge": "Multi-Step",
        "bc": "bc",
        "desc": "LangGraph ReAct loop — agent iterates with tools until it has sufficient context.",
        "doc": f"{_GH}/docs/patterns/02_agentic_rag.md",
    },
    "branched": {
        "icon": "🌿",
        "label": "Branched",
        "badge": "Parallel",
        "bc": "bg",
        "desc": "Decomposes into sub-questions, parallel retrieval, synthesises one answer.",
        "doc": f"{_GH}/docs/patterns/03_branched_rag.md",
    },
    "corrective": {
        "icon": "🔁",
        "label": "CRAG",
        "badge": "Quality Guard",
        "bc": "ba",
        "desc": "Scores retrieved docs; rewrites query or falls back to web search if score is low.",
        "doc": f"{_GH}/docs/patterns/04_corrective_rag.md",
    },
    "graph": {
        "icon": "🕸️",
        "label": "Graph RAG",
        "badge": "Relationships",
        "bc": "bc",
        "desc": "Neo4j knowledge graph traversal augments vector retrieval for entity-hop questions.",
        "doc": f"{_GH}/docs/patterns/05_graph_rag.md",
    },
    "hybrid": {
        "icon": "🔀⚡",
        "label": "Hybrid",
        "badge": "BM25 + Vector",
        "bc": "bg",
        "desc": "BM25 keyword + dense vector search fused via RRF, reranked by cross-encoder. Citation-enforced.",
        "doc": f"{_GH}/docs/patterns/09_hybrid_rag.md",
    },
    "hyde": {
        "icon": "💡",
        "label": "HyDE",
        "badge": "Abstract",
        "bc": "bi",
        "desc": "Generates a hypothetical answer first, embeds it as the search vector (HyDE).",
        "doc": f"{_GH}/docs/patterns/06_hyde_rag.md",
    },
    "multimodal": {
        "icon": "🖼️",
        "label": "Multimodal",
        "badge": "Images + Text",
        "bc": "bg",
        "desc": "Vision LLM describes charts and images alongside text for unified answers.",
        "doc": f"{_GH}/docs/patterns/07_multimodal_rag.md",
    },
    "self_rag": {
        "icon": "🪞",
        "label": "Self-RAG",
        "badge": "Auditable",
        "bc": "br",
        "desc": "Emits [Retrieve][Relevant][Supported][Useful] reflection tokens.",
        "doc": f"{_GH}/docs/patterns/08_self_rag.md",
    },
}

# doc links keyed by strategy id (matches ChunkingStrategy enum values)
CHUNKING_DOCS: dict[str, str] = {
    "recursive": f"{_GH}/docs/chunking/01_recursive.md",
    "character": f"{_GH}/docs/chunking/02_character.md",
    "token": f"{_GH}/docs/chunking/03_token.md",
    "sentence_transformers": f"{_GH}/docs/chunking/04_sentence_transformers.md",
    "semantic": f"{_GH}/docs/chunking/05_semantic.md",
    "sentence_window": f"{_GH}/docs/chunking/06_sentence_window.md",
    "markdown": f"{_GH}/docs/chunking/07_markdown.md",
    "html": f"{_GH}/docs/chunking/08_html.md",
    "parent_child": f"{_GH}/docs/chunking/09_parent_child.md",
    "hypothetical_questions": f"{_GH}/docs/chunking/10_hypothetical_questions.md",
}

# ─────────────────────────────────────────────────────────────────────────────
# API helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _fetch_strategies() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/ingest/strategies", timeout=3)
        return r.json() if r.ok else []
    except Exception:
        return []


@st.cache_data(ttl=20)
def _fetch_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=20)
def _fetch_qdrant_stats() -> dict:
    """Fetch collection stats via the FastAPI /health/qdrant-stats proxy endpoint,
    falling back to a direct Qdrant call using the URL and collection name from
    the API's own settings so we never hardcode either value here."""
    # Primary: ask our own API (avoids duplicating config in the UI layer)
    try:
        r = requests.get(f"{API_BASE}/health/qdrant-stats", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass

    # Fallback: hit Qdrant directly via the API's reported base URL
    try:
        cfg_r = requests.get(f"{API_BASE}/health/config", timeout=3)
        qdrant_url = (
            cfg_r.json().get("qdrant_url", "http://localhost:6333")
            if cfg_r.ok
            else "http://localhost:6333"
        )
        collection = (
            cfg_r.json().get("qdrant_collection", "docustra_docs") if cfg_r.ok else "docustra_docs"
        )
    except Exception:
        qdrant_url, collection = "http://localhost:6333", "docustra_docs"

    try:
        r = requests.get(f"{qdrant_url}/collections/{collection}", timeout=3)
        if r.ok:
            info = r.json().get("result", {})
            return {
                "vectors": info.get("points_count", 0),
                "status": info.get("status", "—"),
                "optimizer": info.get("optimizer_status", {}).get("status", "—"),
            }
        return {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic parameter widgets
# ─────────────────────────────────────────────────────────────────────────────


def _render_params(strategy_info: dict) -> dict:
    params = strategy_info.get("params", [])
    if not params:
        st.markdown(
            '<div style="color:var(--text-muted);font-size:.82rem;padding:10px 0">'
            "No configurable parameters for this strategy.</div>",
            unsafe_allow_html=True,
        )
        return {}

    values: dict = {}
    cols_per_row = 2
    for i in range(0, len(params), cols_per_row):
        batch = params[i : i + cols_per_row]
        cols = st.columns(len(batch))
        for col, p in zip(cols, batch, strict=False):
            with col:
                name, ptype, default = p["name"], p["type"], p["default"]
                sid = strategy_info["id"]
                key = f"p_{sid}_{name}"
                label, help_txt = p["label"], p.get("help", "")

                if ptype == "int":
                    values[name] = int(
                        st.number_input(
                            label,
                            min_value=p.get("min_val") or 1,
                            max_value=p.get("max_val") or 9999,
                            value=int(default),
                            step=1,
                            help=help_txt,
                            key=key,
                        )
                    )
                elif ptype == "float":
                    values[name] = float(
                        st.number_input(
                            label,
                            min_value=float(p.get("min_val") or 0.0),
                            max_value=float(p.get("max_val") or 1.0),
                            value=float(default),
                            step=0.01,
                            help=help_txt,
                            key=key,
                        )
                    )
                elif ptype == "text":
                    values[name] = st.text_input(label, value=str(default), help=help_txt, key=key)
                elif ptype == "select":
                    opts = p.get("options") or []
                    idx = opts.index(default) if default in opts else 0
                    values[name] = st.selectbox(
                        label, options=opts, index=idx, help=help_txt, key=key
                    )
    return values


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Docustra",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero">'
        "<h1>📚 Docustra</h1>"
        "<p>Enterprise Document Intelligence &nbsp;·&nbsp; "
        "9 RAG Patterns &nbsp;·&nbsp; 10 Chunking Strategies &nbsp;·&nbsp; "
        "Hybrid BM25+Vector · Cross-Encoder Reranking · Citation Enforcement · "
        "FastAPI · Qdrant · LangGraph</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ingest, tab_query, tab_system = st.tabs(
        ["📄  Document Intelligence", "🔍  RAG Query", "📊  System Dashboard"]
    )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — INGEST
    # ═════════════════════════════════════════════════════════════════════════
    with tab_ingest:
        strategies = _fetch_strategies()
        smap = {s["id"]: s for s in strategies}
        sids = list(smap) or ["recursive"]

        st.markdown(
            '<div class="ct" style="font-size:.8rem;margin-bottom:4px">INGEST PIPELINE</div>'
            '<p style="color:var(--text-muted);font-size:.88rem;margin:0 0 20px">Select a chunking strategy, '
            "configure its parameters, then index your PDF into Qdrant.</p>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1], gap="large")

        with left:
            uploaded = st.file_uploader(
                "Drop PDF here or click to browse",
                type=["pdf"],
                label_visibility="collapsed",
            )
            if uploaded:
                st.markdown(
                    f'<div style="color:var(--emerald);font-size:.83rem;margin-top:4px">'
                    f"✓ &nbsp;{uploaded.name} &nbsp;·&nbsp; {uploaded.size / 1024:.0f} KB</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="ct">CHUNKING STRATEGY</div>', unsafe_allow_html=True)

            def _label(sid: str) -> str:
                s = smap.get(sid, {})
                return s.get("name", sid).replace("_", " ").title() + (
                    " 🤖" if s.get("requires_llm") else ""
                )

            sel_idx = st.selectbox(
                "strat",
                options=range(len(sids)),
                format_func=lambda i: _label(sids[i]),
                label_visibility="collapsed",
                key="strat_sel",
            )
            sel_id = sids[sel_idx]
            sel = smap.get(sel_id, {})

            if sel:
                req = sel.get("requires_llm", False)
                bc = "ba" if req else "bg"
                bt = "🤖 LLM Required" if req else "⚡ No LLM"
                doc_url = CHUNKING_DOCS.get(sel_id, "")
                doc_link = (
                    f'<a href="{doc_url}" target="_blank" '
                    f'style="display:inline-flex;align-items:center;gap:4px;margin-top:10px;'
                    f"font-size:.76rem;color:var(--primary-soft);text-decoration:none;"
                    f"padding:4px 10px;border:1px solid rgba(129,140,248,.3);border-radius:20px;"
                    f'background:rgba(99,102,241,.08);transition:all .15s">'
                    f"📖 Read full docs ↗</a>"
                    if doc_url
                    else ""
                )
                st.markdown(
                    f'<div style="margin-top:10px;padding:14px 16px;background:var(--surface);'
                    f'border:1px solid var(--border);border-radius:10px">'
                    f'<span class="badge {bc}" style="margin-bottom:8px;display:inline-block">{bt}</span>'
                    f'<div style="font-size:.83rem;color:var(--text);line-height:1.5">{sel.get("description", "")}</div>'
                    f'<div style="font-size:.76rem;color:var(--text-muted);margin-top:6px">'
                    f"Best for: {sel.get('best_for', '')}</div>"
                    f"{doc_link}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if req:
                    st.warning(
                        "LLM called once per chunk — slow on free Gemini tier (15 RPM).", icon="⚠️"
                    )

        with right:
            st.markdown('<div class="ct">PARAMETERS</div>', unsafe_allow_html=True)
            chunking_params = _render_params(sel) if sel else {}

            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="ct">OPTIONS</div>', unsafe_allow_html=True)
            build_graph = st.toggle(
                "Build Knowledge Graph",
                value=False,
                help="Extracts entities → Neo4j. Required for Graph RAG only.",
            )
            if build_graph:
                st.markdown(
                    '<div style="font-size:.78rem;color:var(--amber);margin-top:4px">'
                    "⚠️ Adds 2-5 min for large PDFs on free Gemini tier.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        btn_col, tip_col = st.columns([2, 5])
        with btn_col:
            ingest_btn = st.button(
                "⬆️  Ingest Document",
                type="primary",
                disabled=not uploaded,
                use_container_width=True,
            )
        with tip_col:
            if not uploaded:
                st.markdown(
                    '<div style="color:var(--text-muted);font-size:.82rem;padding-top:10px">'
                    "Upload a PDF above to enable ingestion.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="color:var(--text-muted);font-size:.82rem;padding-top:10px">'
                    f"Ready: <strong style='color:var(--text)'>{uploaded.name}</strong> · "
                    f"Strategy: <strong style='color:var(--primary-soft)'>{_label(sel_id)}</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if ingest_btn and uploaded:
            bar = st.progress(0, text="Starting…")
            t0 = time.time()
            bar.progress(20, text="Parsing PDF…")
            try:
                r = requests.post(
                    f"{API_BASE}/ingest/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    data={
                        "build_graph": str(build_graph).lower(),
                        "chunking_strategy": sel_id,
                        "chunking_params": json.dumps(chunking_params),
                    },
                    timeout=600,
                )
                bar.progress(90, text="Indexing vectors…")
            except requests.exceptions.Timeout:
                bar.empty()
                st.error("Request timed out — the document may be very large.")
                st.stop()

            elapsed = round(time.time() - t0, 1)
            bar.empty()

            if r.status_code == 200:
                d = r.json()
                st.markdown(
                    f'<div style="padding:12px 18px;background:rgba(16,185,129,.1);'
                    f"border:1px solid rgba(16,185,129,.3);border-radius:8px;"
                    f'color:var(--emerald);font-weight:600;font-size:.9rem">'
                    f"✅ Ingested in {elapsed}s</div>",
                    unsafe_allow_html=True,
                )
                st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Chunks", f"{d['chunks_indexed']:,}")
                m2.metric(
                    "Strategy", d.get("chunking_strategy", "recursive").replace("_", " ").title()
                )
                m3.metric("Images", d.get("images_found", 0))
                m4.metric("KG Entities", d.get("graph_entities", 0))
                with st.expander("📋 Full response"):
                    st.json(d)
                _fetch_qdrant_stats.clear()
            else:
                st.error(f"Ingestion failed ({r.status_code})")
                st.code(r.text)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — QUERY
    # ═════════════════════════════════════════════════════════════════════════
    with tab_query:
        st.markdown(
            '<div class="ct">CHOOSE RAG PATTERN</div>',
            unsafe_allow_html=True,
        )

        if "sel_pat" not in st.session_state:
            st.session_state.sel_pat = "adaptive"

        pids = list(RAG_PATTERNS)
        # 3 rows × 3 cols (9 patterns)
        for row_pids in [pids[:3], pids[3:6], pids[6:]]:
            cols = st.columns(3)
            for col, pid in zip(cols, row_pids, strict=False):
                p = RAG_PATTERNS[pid]
                is_sel = st.session_state.sel_pat == pid
                sc = "sel" if is_sel else ""
                doc_url = p.get("doc", "")
                doc_link = (
                    f'<a href="{doc_url}" target="_blank" '
                    f'style="display:inline-flex;align-items:center;gap:3px;margin-top:8px;'
                    f"font-size:.72rem;color:var(--primary-soft);text-decoration:none;"
                    f"padding:3px 8px;border:1px solid rgba(129,140,248,.25);border-radius:20px;"
                    f'background:rgba(99,102,241,.07)">📖 Docs ↗</a>'
                    if doc_url
                    else ""
                )
                col.markdown(
                    f'<div class="pcard {sc}">'
                    f'<div class="pi">{p["icon"]}</div>'
                    f'<div class="pn">{p["label"]}</div>'
                    f'<div style="margin:4px 0"><span class="badge {p["bc"]}">{p["badge"]}</span></div>'
                    f'<div class="pd">{p["desc"]}</div>'
                    f"{doc_link}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if col.button(
                    "✓ Select" if is_sel else "Select",
                    key=f"pb_{pid}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.sel_pat = pid
                    st.rerun()

        cp_id = st.session_state.sel_pat
        cp = RAG_PATTERNS[cp_id]
        cp_doc = cp.get("doc", "")
        cp_doc_link = (
            f' &nbsp;<a href="{cp_doc}" target="_blank" '
            f'style="font-size:.74rem;color:var(--primary-soft);text-decoration:none;'
            f"padding:3px 9px;border:1px solid rgba(129,140,248,.25);border-radius:20px;"
            f'background:rgba(99,102,241,.07)">📖 Read docs ↗</a>'
            if cp_doc
            else ""
        )
        st.markdown(
            f'<div style="margin:16px 0 8px;font-size:.84rem;color:var(--text-muted);'
            f'display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'Active: {cp["icon"]} <strong style="color:var(--text)">{cp["label"]}</strong>'
            f' &nbsp;<span class="badge {cp["bc"]}">{cp["badge"]}</span>'
            f"{cp_doc_link}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="ct" style="margin-top:16px">YOUR QUESTION</div>', unsafe_allow_html=True
        )
        qcol, bcol = st.columns([7, 1])
        with qcol:
            question = st.text_area(
                "q",
                label_visibility="collapsed",
                placeholder="e.g. What are the main differences between HNSW and IVF indexing?",
                height=96,
            )
        with bcol:
            st.markdown('<div style="height:38px"></div>', unsafe_allow_html=True)
            query_btn = st.button(
                "🔎 Query",
                type="primary",
                disabled=not question.strip(),
                use_container_width=True,
            )

        if query_btn and question.strip():
            with st.spinner(f"{cp['icon']} Running **{cp['label']}** RAG…"):
                t0 = time.time()
                try:
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={"question": question, "pattern": cp_id},
                        timeout=180,
                    )
                except requests.exceptions.Timeout:
                    st.error("Query timed out.")
                    st.stop()
            elapsed = round(time.time() - t0, 1)

            if r.status_code == 200:
                d = r.json()
                st.divider()

                # ── Declined / cannot answer banner ──────────────────────────
                answer_text = d["answer"]
                is_declined = answer_text.startswith("I cannot answer")
                if is_declined:
                    st.warning(
                        "⚠️ **Insufficient context** — the model declined to answer because "
                        "the retrieved documents do not contain sufficient information. "
                        "Try ingesting more relevant documents or rephrasing your question.",
                        icon=None,
                    )

                st.markdown('<div class="ct">ANSWER</div>', unsafe_allow_html=True)
                acol, scol = st.columns([4, 1])
                with acol:
                    st.markdown(
                        f'<div style="font-size:.95rem;line-height:1.7;color:var(--text)">{answer_text}</div>',
                        unsafe_allow_html=True,
                    )
                with scol:
                    st.metric("Latency", f"{elapsed}s")
                    st.metric("Sources", len(d.get("sources", [])))
                    citations = d.get("citations", [])
                    if citations:
                        st.metric("Citations", len(citations))

                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

                citations = d.get("citations", [])
                num_expanders = 4 if citations else 3
                cols_exp = st.columns(num_expanders)

                with cols_exp[0], st.expander("🧠 Reasoning & Reflection"):
                    if d.get("reasoning"):
                        st.code(d["reasoning"], language=None)
                    else:
                        st.markdown(
                            '<span style="color:var(--text-muted);font-size:.82rem">'
                            "No reasoning for this pattern.</span>",
                            unsafe_allow_html=True,
                        )

                with cols_exp[1], st.expander(f"📎 Sources ({len(d.get('sources', []))})"):
                    for s in d.get("sources", []):
                        st.markdown(
                            f"**{s.get('source', '?').split('/')[-1]}** — Page {s.get('page', '?')}"
                        )
                        st.caption(s.get("content", "")[:200])
                        st.divider()

                if citations:
                    with cols_exp[2], st.expander(f"🔖 Citations ({len(citations)})"):
                        for i, c in enumerate(citations, start=1):
                            src = c.get("source", "unknown").split("/")[-1]
                            page = c.get("page")
                            score = c.get("reranker_score")
                            header = f"**[{i}] {src}**"
                            if page:
                                header += f" — Page {page}"
                            if score is not None:
                                header += f" *(relevance: {score:.3f})*"
                            st.markdown(header)
                            st.caption(c.get("passage_preview", "")[:200])
                            st.divider()

                with cols_exp[-1], st.expander("🔧 Raw Metadata"):
                    st.json(d.get("metadata", {}))
            else:
                st.error(f"Query failed ({r.status_code})")
                st.code(r.text)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — SYSTEM DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    with tab_system:
        rc, _ = st.columns([1, 6])
        with rc:
            if st.button("🔄 Refresh", key="dash_ref"):
                _fetch_health.clear()
                _fetch_qdrant_stats.clear()
                st.rerun()

        # Service health
        st.markdown(
            '<div class="ct" style="margin-bottom:12px">SERVICE HEALTH</div>',
            unsafe_allow_html=True,
        )
        health = _fetch_health()
        svc_meta = {
            "qdrant": ("🗄️", "Vector Store", "http://localhost:6333/dashboard"),
            "neo4j": ("🕸️", "Knowledge Graph", "http://localhost:7474"),
            "redis": ("⚡", "Cache / Queue", "redis://localhost:6379"),
        }
        if health:
            scols = st.columns(3)
            for col, (svc, status) in zip(scols, health.get("services", {}).items(), strict=False):
                icon, desc, url = svc_meta.get(svc, ("🔧", svc.title(), "#"))
                ok = status == "ok"
                dot = "var(--emerald)" if ok else "var(--red)"
                col.markdown(
                    f'<div class="svc">'
                    f'<div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>'
                    f'<div style="font-weight:700;font-size:.9rem;color:var(--text)">{svc.title()}</div>'
                    f'<div style="font-size:.76rem;color:var(--text-muted);margin-bottom:10px">{desc}</div>'
                    f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<span style="color:{dot}">●</span>'
                    f'<span style="font-size:.82rem;color:{dot}">{"Healthy" if ok else "Unreachable"}</span>'
                    f"</div>"
                    f'<a href="{url}" target="_blank" style="font-size:.74rem;color:var(--primary-soft);'
                    f'text-decoration:none;display:block;margin-top:8px">Open ↗</a>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Cannot reach API server at http://localhost:8000")

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # Qdrant stats
        st.markdown(
            '<div class="ct" style="margin-bottom:12px">QDRANT COLLECTION — docustra_docs</div>',
            unsafe_allow_html=True,
        )
        stats = _fetch_qdrant_stats()
        if stats:
            s1, s2, s3 = st.columns(3)
            s1.metric("Vectors Stored", f"{stats.get('vectors', 0):,}")
            s2.metric("Collection Status", stats.get("status", "—").title())
            s3.metric("Optimizer", stats.get("optimizer", "—").title())
            st.markdown(
                '<div style="font-size:.78rem;color:var(--text-muted);margin-top:8px">'
                '🔗 <a href="http://localhost:6333/dashboard" target="_blank" style="color:var(--primary-soft)">Qdrant Dashboard</a>'
                " &nbsp;·&nbsp; "
                '🔗 <a href="http://localhost:6006" target="_blank" style="color:var(--primary-soft)">Arize Phoenix Traces</a>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Qdrant not reachable or collection not yet created.")

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

        # Chunking reference
        st.markdown(
            '<div class="ct" style="margin-bottom:12px">CHUNKING STRATEGY REFERENCE</div>',
            unsafe_allow_html=True,
        )
        strats = _fetch_strategies()
        if strats:
            g1, g2 = st.columns(2)
            for i, s in enumerate(strats):
                col = g1 if i % 2 == 0 else g2
                req = s.get("requires_llm", False)
                bc = "ba" if req else "bg"
                bt = "🤖 LLM" if req else "⚡ Fast"
                pp = " · ".join(f"{p['name']}={p['default']}" for p in s.get("params", [])[:2])
                doc_url = CHUNKING_DOCS.get(s["id"], "")
                doc_link = (
                    f'<a href="{doc_url}" target="_blank" '
                    f'style="display:inline-flex;align-items:center;gap:3px;margin-top:10px;'
                    f"font-size:.73rem;color:var(--primary-soft);text-decoration:none;"
                    f"padding:4px 10px;border:1px solid rgba(129,140,248,.28);border-radius:20px;"
                    f'background:rgba(99,102,241,.08)">📖 Read docs ↗</a>'
                    if doc_url
                    else ""
                )
                col.markdown(
                    f'<div class="card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:6px">'
                    f'<div style="font-weight:700;font-size:.88rem;color:var(--text)">{s["name"]}</div>'
                    f'<span class="badge {bc}">{bt}</span>'
                    f"</div>"
                    f'<div style="font-size:.78rem;color:var(--text-muted);line-height:1.4;margin-bottom:6px">{s["description"]}</div>'
                    f'<div style="font-size:.74rem;color:var(--text-muted)">'
                    f'<span style="color:var(--primary-soft)">Best for:</span> {s["best_for"]}</div>'
                    + (
                        f'<div style="margin-top:6px;font-size:.72rem;color:var(--border-glow)">Defaults: <code>{pp}</code></div>'
                        if pp
                        else ""
                    )
                    + doc_link
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Could not load strategies — is the API running?")

        # Documentation index
        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ct" style="margin-bottom:12px">DOCUMENTATION</div>', unsafe_allow_html=True
        )

        doc_links = [
            (
                "📚",
                "Full Project README",
                f"{_GH}/README.md",
                "Project overview, setup, quick start",
            ),
            (
                "🔍",
                "RAG Patterns Overview",
                f"{_GH}/docs/patterns/00_overview.md",
                "Comparison table, decision guide, latency benchmarks",
            ),
            (
                "✂️",
                "Chunking Strategies Overview",
                f"{_GH}/docs/chunking/00_overview.md",
                "All 10 strategies, decision guide, chunk size tuning",
            ),
            (
                "📸",
                "UI Screenshots",
                f"{_GH}/docs/screenshots/README.md",
                "Screenshot guide for portfolio showcase",
            ),
        ]

        d_cols = st.columns(4)
        for d_col, (icon, title, url, hint) in zip(d_cols, doc_links, strict=False):
            d_col.markdown(
                f'<a href="{url}" target="_blank" style="text-decoration:none">'
                f'<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;'
                f'padding:14px 16px;transition:border-color .2s;height:100%;cursor:pointer" '
                f"onmouseover=\"this.style.borderColor='#6366F1'\" "
                f"onmouseout=\"this.style.borderColor='#2D3748'\">"
                f'<div style="font-size:1.4rem;margin-bottom:6px">{icon}</div>'
                f'<div style="font-weight:600;font-size:.84rem;color:var(--text);margin-bottom:4px">{title}</div>'
                f'<div style="font-size:.74rem;color:var(--text-muted);line-height:1.4">{hint}</div>'
                f'<div style="margin-top:8px;font-size:.72rem;color:var(--primary-soft)">Open on GitHub ↗</div>'
                f"</div></a>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(
            '<div style="text-align:center;color:var(--text-muted);font-size:.78rem">'
            "Docustra &nbsp;·&nbsp; "
            '<a href="https://github.com/aritraju/docustra" target="_blank" style="color:var(--primary-soft)">GitHub</a>'
            " &nbsp;·&nbsp; "
            '<a href="http://localhost:8000/docs" target="_blank" style="color:var(--primary-soft)">API Docs</a>'
            " &nbsp;·&nbsp; "
            '<a href="http://localhost:6006" target="_blank" style="color:var(--primary-soft)">Phoenix Traces</a>'
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
