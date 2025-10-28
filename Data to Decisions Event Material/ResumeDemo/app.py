# app.py
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Class Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Minimal CSS polish ----------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(66, 66, 66, 0.35), rgba(66, 66, 66, 0.10));
        border: 1px solid rgba(255,255,255,0.06);
    }
    .card {
        padding: 1.15rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
    }
    .kpi {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.35rem;
    }
    .pill {
        display:inline-block;
        padding: .2rem .55rem;
        font-size: .8rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.04);
        margin-right: .35rem;
        margin-top: .35rem;
    }
    .small {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Title / Hero ----------
st.markdown(
    """
    <h1 style="display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;">
      <span style="font-weight:800; font-size:2.4rem; color:#8A2BE2;">K</span>
      <span>Class Explorer</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h3 style="margin-top:0">Explore your class from every angle</h3>
      <p class="small">
        Use the sidebar to open each tool. These pages read the processed outputs you generated
        (summaries and the enriched records) and give you fast, filterable views—plus a semantic
        Q&A experience grounded on the full resume set.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Quick file status ----------
summaries_path = Path("summaries.json")
enriched_jsonl = Path("resumes_enriched.jsonl")
index_file = Path("enriched_index.npz")

colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**📄 summaries.json**")
    st.markdown(
        f"<div class='kpi'>{'✅ Found' if summaries_path.exists() else '⚠️ Not found'} — used by the map & skill matcher</div>",
        unsafe_allow_html=True,
    )
with colB:
    st.markdown("**🧱 resumes_enriched.jsonl**")
    st.markdown(
        f"<div class='kpi'>{'✅ Found' if enriched_jsonl.exists() else '⚠️ Not found'} — used by semantic search</div>",
        unsafe_allow_html=True,
    )
with colC:
    st.markdown("**🗂️ enriched_index.npz**")
    st.markdown(
        f"<div class='kpi'>{'✅ Found' if index_file.exists() else 'ℹ️ Will be built on first use'} — vector cache for semantic search</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Product Cards ----------
c1, c2 = st.columns([1,1])
c3, _  = st.columns([1,1])

with c1:
    st.markdown("### 🗺️ Class Map")
    st.markdown(
        """
        <div class="card">
        <p class="small">
        Explore students by **location**. Quickly answer questions like:
        <br>• *“Who’s in Austin or NYC?”*<br>
        • *“Which cities have the biggest clusters?”*
        </p>
        <div class="small">
          <span class="pill">Reads: <code>summaries.json</code></span>
          <span class="pill">Filters: Location</span>
          <span class="pill">Output: Interactive map + table</span>
        </div>
        <p class="small" style="margin-top:.7rem;">
          <strong>Best for:</strong> recruiting roadmaps, meetup planning, and geo-targeted outreach.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("### 🎯 Skill Matcher")
    st.markdown(
        """
        <div class="card">
        <p class="small">
        Search by **skills**, **experience type**, **employers**, or **name**. Build shortlists like:
        <br>• *“PMs with healthcare experience in Chicago”*<br>
        • *“Spanish speakers interested in climate tech”*
        </p>
        <div class="small">
          <span class="pill">Reads: <code>summaries.json</code></span>
          <span class="pill">Filters: Skills, Experience, Name, Locations</span>
          <span class="pill">Output: Ranked, filterable table</span>
        </div>
        <p class="small" style="margin-top:.7rem;">
          <strong>Best for:</strong> team staffing, club recruiting, and targeted networking.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("### 🧠 Semantic Search")
    st.markdown(
        """
        <div class="card">
        <p class="small">
        Ask free-form questions over the **full resume corpus**. The app retrieves the most relevant
        profiles and uses a local LLM (Gemma via Ollama) to craft answers grounded in those snippets.
        <br><br>
        Examples:
        <br>• *“How many people worked at Deloitte?”* (fast-path count, no LLM)  
        • *“Who has consulting experience in Texas?”*  
        • *“Which students mention Python & healthcare?”*
        </p>
        <div class="small">
          <span class="pill">Reads: <code>resumes_enriched.jsonl</code></span>
          <span class="pill">Vector cache: <code>enriched_index.npz</code></span>
          <span class="pill">Models: nomic-embed-text / MiniLM • Gemma3:1B (Ollama)</span>
        </div>
        <p class="small" style="margin-top:.7rem;">
          <strong>Notes:</strong> employer/college/skill/location filters pre-narrow results;
          count/has/who-at-Company questions run deterministically for speed.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------- Getting Started / Tips ----------
st.markdown("### 🚀 Getting started")
st.markdown(
    """
    1. **Generate inputs**  
       • `summaries.json` from your parsing/summary step  
       • `resumes_enriched.jsonl` from the enrichment step (classification, skills, interests, etc.)  
    2. **Run the app**: `streamlit run app.py`  
    3. **Open a page** from the sidebar and start exploring.
    """,
)

st.markdown("### 🧩 Model & performance tips")
st.markdown(
    """
    - **Ollama running?** Start it and pull models:  
      `ollama serve` · `ollama pull gemma3:1b` · `ollama pull nomic-embed-text`  
    - Semantic Search caps snippets for the LLM and has a **fast-path** for company counts (e.g., *“How many worked at PwC?”*).  
    - If embeddings aren’t available in Ollama, the app falls back to **Sentence-Transformers** (MiniLM).
    """,
)

st.caption("© Kellogg Data and Analytics Club — Class Explorer")
