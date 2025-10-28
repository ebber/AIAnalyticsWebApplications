# pages/02_Skill_Matcher.py
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

st.set_page_config(page_title="Skill Matcher", layout="wide")

ENRICHED_PATH = Path("resumes_enriched.jsonl")  # step-2/3 output
SUMMARIES_PATH = Path("summaries.json")         # fallback

# ---------------- Utilities ----------------
def _join_list(x):
    return ", ".join(x) if isinstance(x, list) else (x or "")

@st.cache_data
def load_records():
    """
    Prefer the enriched JSONL; fallback to summaries.json.
    Always return Series (never raw strings) so .fillna works.
    """
    rows = []
    used = None

    if ENRICHED_PATH.exists():
        used = "enriched"
        with ENRICHED_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif SUMMARIES_PATH.exists():
        used = "summaries"
        df = pd.DataFrame(json.loads(SUMMARIES_PATH.read_text(encoding="utf-8")))
    else:
        st.error("No data found. Expected `resumes_enriched.jsonl` or `summaries.json`.")
        st.stop()

    # --- helpers ---
    def col_or_empty(names, *, listy=False):
        """Return a Series for the first existing name; else empty Series of proper length."""
        for n in names:
            if n in df.columns:
                s = df[n]
                # If the upstream is not a Series (rare), coerce
                if not isinstance(s, pd.Series):
                    s = pd.Series(s)
                return s
        # Make an empty series matching df length
        if listy:
            return pd.Series([[] for _ in range(len(df))])
        else:
            return pd.Series(["" for _ in range(len(df))])

    def to_list_series(s):
        """Ensure each cell is a list (handles None/NaN/str)."""
        return s.apply(lambda x: x if isinstance(x, list) else ([] if (x is None or (isinstance(x, float) and pd.isna(x))) else ([x] if isinstance(x, str) and x else [])))

    # --- normalize canonical fields ---
    name_s        = col_or_empty(["name_extracted", "Name"]).fillna("")
    exptype_s     = col_or_empty(["experience_type", "Experience Type"]).fillna("")
    locs_s        = to_list_series(col_or_empty(["locations_extracted", "Location(s)"], listy=True))
    undergrad_s   = to_list_series(col_or_empty(["undergrad_extracted", "Undergrad Institution(s)"], listy=True))
    skills_s      = to_list_series(col_or_empty(["skills", "Skills"], listy=True))
    employers_s   = to_list_series(col_or_empty(["employers", "Employers"], listy=True))
    roles_s       = to_list_series(col_or_empty(["roles_extracted", "Roles"], listy=True))
    rawtext_s     = col_or_empty(["raw_text"]).fillna("")

    out = pd.DataFrame({
        "Name": name_s.astype(str),
        "ExpType": exptype_s.astype(str),
        "Locs_list": locs_s,
        "Undergrad_list": undergrad_s,
        "Skills_list": skills_s,
        "Employers_list": employers_s,
        "Roles_list": roles_s,
        "RawText": rawtext_s.astype(str),
    })

    # stringified versions
    def join_list(x): return ", ".join(x) if isinstance(x, list) else (x or "")
    out["Skills_str"]    = out["Skills_list"].apply(join_list)
    out["Employers_str"] = out["Employers_list"].apply(join_list)
    out["Roles_str"]     = out["Roles_list"].apply(join_list)
    out["Locs_str"]      = out["Locs_list"].apply(join_list)
    out["Undergrad_str"] = out["Undergrad_list"].apply(join_list)
    out["Text_str"]      = out["RawText"].str.replace("\u00a0", " ", regex=False).str[:4000]

    # lowercased blobs for scoring
    out["Blob_skills"] = out["Skills_str"].str.lower()
    out["Blob_roles"]  = (out["Roles_str"] + " | " + out["Employers_str"]).str.lower()
    out["Blob_meta"]   = (out["Name"] + " | " + out["ExpType"] + " | " + out["Undergrad_str"] + " | " + out["Locs_str"]).str.lower()
    out["Blob_text"]   = out["Text_str"].str.lower()

    return out, used

def tokenize(q: str):
    q = re.sub(r"[^a-z0-9+/#& .-]", " ", q.lower())
    toks = [t for t in re.split(r"\s+", q) if t]
    return toks

def keyword_score(tokens, text_lower: str):
    if not tokens: return 0.0
    hits = sum(1 for t in tokens if t in text_lower)
    return hits / len(tokens)

def fuzzy_score(query: str, text_lower: str):
    # For very short queries (<=3 chars), fuzzy gets noisy — clamp to 0
    if len(query.strip()) <= 3:
        return 0.0
    return fuzz.partial_ratio(query, text_lower) / 100.0

def highlight(text: str, query_tokens):
    if not text: return ""
    out = text
    for t in sorted(set(query_tokens), key=len, reverse=True):
        try:
            out = re.sub(rf"(?i)\b({re.escape(t)})\b", r"**\1**", out)
        except re.error:
            pass
    return out

def word_boundary_mask(series: pd.Series, term: str) -> pd.Series:
    pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I)
    return series.apply(lambda s: bool(pat.search(s or "")))

# ---------------- UI ----------------
st.title("🔎 Skill Matcher — enriched")

df, used_source = load_records()
if used_source == "summaries":
    st.warning("Using fallback `summaries.json`. For best results (roles, richer search), generate and place `resumes_enriched.jsonl`.", icon="⚠️")

st.caption("Search across skills, roles, employers, names, locations, undergrad — optionally include full resume text for deeper matches.")

with st.form("skill_search_form", clear_on_submit=False):
    q = st.text_input(
        "Enter some skills / experiences / names and press Search",
        value="",
        placeholder="e.g., CPA, Python healthcare, Deloitte, private equity Austin, product analytics SQL",
    )

    st.markdown("#### Filters (optional)")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sel_exp = st.multiselect(
            "Experience Type",
            sorted([x for x in df["ExpType"].dropna().unique().tolist() if x]),
        )
    with c2:
        city_filter = st.text_input("Location contains", "")
    with c3:
        undergrad_filter = st.text_input("Undergrad contains", "")

    c4, c5, c6 = st.columns([1,1,1])
    with c4:
        search_only_skills = st.checkbox("Search only in Skills", value=False)
    with c5:
        include_roles = st.checkbox("Include Roles/Employers in search", value=True)
    with c6:
        include_text = st.checkbox("Include Resume Text in search", value=False)

    with st.expander("Scoring & limits"):
        colA, colB, colC = st.columns(3)
        with colA:
            w_skills = st.slider("Weight: Skills", 0.0, 2.0, 1.3, 0.1)
        with colB:
            w_roles  = st.slider("Weight: Roles/Employers", 0.0, 2.0, 1.0, 0.1)
        with colC:
            w_meta   = st.slider("Weight: Name/Exp/Undergrad/Location", 0.0, 2.0, 0.6, 0.1)

        colD, colE, colF = st.columns(3)
        with colD:
            w_text   = st.slider("Weight: Resume Text", 0.0, 2.0, 0.5, 0.1)
        with colE:
            min_score = st.slider("Min overall score", 0.0, 1.0, 0.2, 0.05)
        with colF:
            topk = st.slider("Show top N results", 5, 300, 75, 5)

        st.caption("Tip: Start with Roles ON and Text OFF for speed; turn Text ON only when you need deeper matches.")

    # Strict matching helpers (avoid noise on short tokens)
    c7, c8, _ = st.columns([1,1,1])
    with c7:
        strict_short_tokens = st.checkbox("Strict exact match for short tokens (≤3 chars)", value=True)
    with c8:
        strict_mode = st.checkbox("Strict exact match (all tokens in Skills/Roles/Name/Undergrad)", value=False)

    submitted = st.form_submit_button("Search")

if not submitted and not q.strip():
    st.info("Type a query above and press **Search** to get matches.")
    st.stop()

if not q.strip():
    st.warning("Please enter a query in the search bar.")
    st.stop()

# ---- Apply facet filters first ----
mask = pd.Series(True, index=df.index)
if sel_exp:
    mask &= df["ExpType"].isin(sel_exp)
if city_filter.strip():
    mask &= df["Locs_str"].str.contains(city_filter.strip(), case=False, na=False)
if undergrad_filter.strip():
    mask &= df["Undergrad_str"].str.contains(undergrad_filter.strip(), case=False, na=False)
df_f = df[mask].copy()
if df_f.empty:
    st.warning("No rows match the current filters. Clear some filters and try again.")
    st.stop()

# ---- Strict/short-token gating ----
q_tokens = tokenize(q)
ql = " ".join(q_tokens)

def exact_any_mask(frame: pd.DataFrame, term: str) -> pd.Series:
    fields = ["Skills_str", "Roles_str", "Name", "Undergrad_str"]
    m = pd.Series(False, index=frame.index)
    for f in fields:
        m |= word_boundary_mask(frame[f], term)
    return m

# Only-skills (before scoring)
if search_only_skills and q_tokens:
    m_sk = pd.Series(False, index=df_f.index)
    for t in q_tokens:
        m_sk |= word_boundary_mask(df_f["Skills_str"], t)
    df_f = df_f[m_sk]
    if df_f.empty:
        st.warning("No results where Skills contain those terms.")
        st.stop()

# Strict modes
if strict_mode and q_tokens:
    m_exact = pd.Series(True, index=df_f.index)
    for t in q_tokens:
        m_exact &= exact_any_mask(df_f, t)
    df_f = df_f[m_exact]
    if df_f.empty:
        st.warning("Strict mode removed all results. Try disabling it or broadening the query.")
        st.stop()

if (len(q_tokens) == 1) and (len(q_tokens[0]) <= 3) and strict_short_tokens:
    t = q_tokens[0]
    m_short = exact_any_mask(df_f, t)
    df_f = df_f[m_short]
    if df_f.empty:
        st.warning("No exact matches for that short token in Skills/Roles/Name/Undergrad.")
        st.stop()

# ---- Scoring over filtered set ----
# Choose which blobs to include
use_roles = include_roles
use_text  = include_text

scores = []
for i, row in df_f.iterrows():
    # per-component keyword coverage (and optional fuzzy on meta)
    k_skills = keyword_score(q_tokens, row["Blob_skills"])
    k_roles  = keyword_score(q_tokens, row["Blob_roles"]) if use_roles else 0.0
    k_meta   = keyword_score(q_tokens, row["Blob_meta"])
    k_text   = keyword_score(q_tokens, row["Blob_text"]) if use_text else 0.0

    # Optional fuzzy on meta only (keeps compute light & useful for names/companies)
    f_meta = fuzzy_score(ql, row["Blob_meta"])

    overall = (
        w_skills * k_skills +
        w_roles  * k_roles  +
        w_meta   * max(k_meta, f_meta) +
        w_text   * k_text
    )
    scores.append((i, overall, k_skills, k_roles, k_meta, k_text))

scored = (
    pd.DataFrame(scores, columns=["idx","score","skills","roles","meta","text"])
    .sort_values("score", ascending=False)
)
scored = scored[scored["score"] >= min_score].head(topk)

if scored.empty:
    st.warning("No matches above the minimum score. Lower the threshold or broaden your query.")
    st.stop()

# ---- Compose results
rows = []
for _, r in scored.iterrows():
    row = df_f.loc[r["idx"]]
    rows.append({
        "Score": round(float(r["score"]), 3),
        "Name": row["Name"],
        "Experience Type": row["ExpType"],
        "Skills": row["Skills_str"],
        "Roles": row["Roles_str"],
        "Employers": row["Employers_str"],
        "Locations": row["Locs_str"],
        "Undergrad": row["Undergrad_str"],
    })
out = pd.DataFrame(rows)

st.subheader("Matches")
st.caption(
    "This view searches enriched data. Toggle **Roles/Employers** and **Resume Text** above to balance precision vs depth."
)

view = st.radio("View as", ["Cards", "Table"], horizontal=True, index=0)

def render_card(r):
    st.markdown(f"### {r['Name']} — {r['Experience Type']}  *(score {r['Score']})*")
    st.markdown(f"**Skills:** {highlight(r['Skills'], q_tokens) or '—'}")
    if include_roles:
        st.markdown(f"**Roles / Employers:** {highlight((r['Roles'] or '') + ('; ' + r['Employers'] if r['Employers'] else ''), q_tokens) or '—'}")
    else:
        if r['Employers']:
            st.markdown(f"**Employers:** {highlight(r['Employers'], q_tokens)}")
    st.markdown(f"**Locations:** {r['Locations'] or '—'}")
    st.markdown(f"**Undergrad:** {highlight(r['Undergrad'], q_tokens) or '—'}")
    st.markdown("---")

if view == "Cards":
    for _, row in out.iterrows():
        render_card(row)
else:
    cols = ["Score","Name","Experience Type","Skills"]
    if include_roles:
        cols += ["Roles","Employers"]
    else:
        cols += ["Employers"]
    cols += ["Locations","Undergrad"]
    st.dataframe(out[cols], use_container_width=True, height=600)

st.download_button(
    "Download results (CSV)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="skill_matches_enriched.csv",
    mime="text/csv",
)
