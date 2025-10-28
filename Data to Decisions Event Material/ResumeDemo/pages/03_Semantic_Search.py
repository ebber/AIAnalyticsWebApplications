# pages/03_Semantic_Search.py
import json, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------- CONFIG ---------------------
ENRICHED_JSONL = Path("resumes_enriched.jsonl")   # output from your step-2
INDEX_FILE     = Path("enriched_index.npz")       # cached embeddings
METADATA_JSON  = Path("enriched_metadata.json")

EMBED_MODEL_OLLAMA = "nomic-embed-text"                 # if ollama embed exists
EMBED_MODEL_PY     = "sentence-transformers/all-MiniLM-L6-v2"  # python fallback
GEN_MODEL          = "gemma3:1b"                        # for answering
MAX_CHUNK_CHARS    = 1600

st.set_page_config(page_title="Semantic Search (RAG)", layout="wide")
st.title("🧠 Semantic Search over Resumes (RAG)")

# --------------------- UTILITIES ---------------------
def run_cmd(cmd, stdin=None, timeout=None):
    res = subprocess.run(
        cmd,
        input=(stdin.encode("utf-8") if isinstance(stdin, str) else stdin),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    out = res.stdout.decode("utf-8", errors="replace")
    err = res.stderr.decode("utf-8", errors="replace")
    return res.returncode, out, err

def has_ollama():
    code, _, _ = run_cmd(["ollama", "list"])
    return code == 0

def has_ollama_embed():
    code, out, err = run_cmd(["ollama", "--help"])
    return "embed" in (out + err).lower()

def ensure_gen_model():
    if not has_ollama():
        st.error("`ollama` not found or not running. Start it and pull the generation model:")
        st.code("ollama serve\nollama pull gemma3:1b")
        st.stop()

# --------------------- LOAD & PREP DATA ---------------------
def _as_list(series: pd.Series) -> pd.Series:
    # Normalize each cell to a list
    def norm(x):
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
            return []
        return [str(x)]
    return series.apply(norm)

def _uniq(items: list) -> list:
    seen, out = set(), []
    for it in (items or []):
        s = str(it).strip()
        if not s:
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

@st.cache_data(show_spinner=True)
def load_enriched(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing {path}. Run step 2 to produce it.")
        st.stop()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Ensure expected columns exist
    for col in [
        "resume_id","name_extracted","raw_text","roles_extracted",
        "undergrad_extracted","locations_extracted","experience_type",
        "employers_extracted","skills_rule","skills_llm","interests_rule","interests_llm"
    ]:
        if col not in df.columns:
            df[col] = None

    # Canonical/simple columns
    df["name"]            = df["name_extracted"].fillna("")
    df["experience_type"] = df["experience_type"].fillna("")
    df["raw_text"]        = df["raw_text"].fillna("")

    # Normalize and merge lists
    skills_llm  = _as_list(df["skills_llm"])
    skills_rule = _as_list(df["skills_rule"])
    ints_llm    = _as_list(df["interests_llm"])
    ints_rule   = _as_list(df["interests_rule"])

    df["skills_list"]    = (skills_llm + skills_rule).apply(_uniq)
    df["interests_list"] = (ints_llm + ints_rule).apply(_uniq)
    df["employers_list"] = _as_list(df["employers_extracted"]).apply(_uniq)
    df["colleges_list"]  = _as_list(df["undergrad_extracted"]).apply(_uniq)
    df["locations_list"] = _as_list(df["locations_extracted"]).apply(_uniq)

    # Retrieval chunk with all structured fields
    def mk_chunk(r):
        parts = []
        if r.get("name"): parts.append(f"NAME: {r['name']}")
        if r.get("experience_type"): parts.append(f"EXPERIENCE_TYPE: {r['experience_type']}")
        if r.get("roles_extracted"): parts.append("ROLES: " + ", ".join(r.get("roles_extracted") or []))
        if r.get("employers_list"): parts.append("EMPLOYERS: " + ", ".join(r["employers_list"]))
        if r.get("colleges_list"):  parts.append("UNDERGRAD: " + ", ".join(r["colleges_list"]))
        if r.get("skills_list"):    parts.append("SKILLS: " + ", ".join(r["skills_list"]))
        if r.get("interests_list"): parts.append("INTERESTS: " + ", ".join(r["interests_list"]))
        if r.get("locations_list"): parts.append("LOCATIONS: " + ", ".join(r["locations_list"]))
        raw = (r.get("raw_text") or "").replace("\u00a0", " ")
        parts.append("TEXT: " + raw[:MAX_CHUNK_CHARS])
        return "\n".join(parts)

    df["chunk"] = df.apply(mk_chunk, axis=1)

    # Lowercase blobs for filtering
    df["raw_lower"]      = df["raw_text"].str.lower()
    df["employers_blob"] = df["employers_list"].apply(lambda xs: " | ".join(xs).lower())
    df["colleges_blob"]  = df["colleges_list"].apply(lambda xs: " | ".join(xs).lower())
    df["skills_blob"]    = df["skills_list"].apply(lambda xs: " | ".join(xs).lower())
    df["interests_blob"] = df["interests_list"].apply(lambda xs: " | ".join(xs).lower())
    df["locations_blob"] = df["locations_list"].apply(lambda xs: " | ".join(xs).lower())
    return df

# --------------------- EMBEDDINGS ---------------------
def embed_ollama(texts: list[str]) -> np.ndarray:
    payload = json.dumps({"input": texts})
    code, out, err = run_cmd(["ollama", "embed", "-m", EMBED_MODEL_OLLAMA], stdin=payload)
    if code != 0:
        raise RuntimeError(f"Ollama embed failed: {err or out}")
    obj = json.loads(out)
    return np.array(obj["embeddings"], dtype=np.float32)

def load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        st.error("Embedding fallback requires:\n\npip install sentence-transformers")
        st.stop()
    return SentenceTransformer(EMBED_MODEL_PY)

@st.cache_resource(show_spinner=True)
def build_or_load_index(df: pd.DataFrame):
    backend = "ollama" if (has_ollama() and has_ollama_embed()) else "py"

    if INDEX_FILE.exists() and METADATA_JSON.exists():
        try:
            data = np.load(INDEX_FILE)
            E = data["embeddings"]
            meta = json.loads(METADATA_JSON.read_text(encoding="utf-8"))
            if len(meta) == len(df):
                return backend, E, meta
        except Exception:
            pass

    st.info("Building the vector index (one-time)…")
    chunks = df["chunk"].tolist()
    if backend == "ollama":
        run_cmd(["ollama", "pull", EMBED_MODEL_OLLAMA])
        E = embed_ollama(chunks)
    else:
        model = load_sentence_transformer()
        E = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)

    meta = []
    for _, r in df.iterrows():
        meta.append({
            "resume_id": r.get("resume_id"),
            "name": r.get("name"),
            "experience_type": r.get("experience_type"),
            "locations": r.get("locations_list") or [],
        })
    np.savez_compressed(INDEX_FILE, embeddings=E)
    METADATA_JSON.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return backend, E, meta

def embed_query(q: str, backend: str) -> np.ndarray:
    if backend == "ollama":
        return embed_ollama([q])
    else:
        model = load_sentence_transformer()
        return model.encode([q], normalize_embeddings=True).astype(np.float32)

def cosine_search(E: np.ndarray, q_emb: np.ndarray, top_k: int = 10):
    if E.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    A = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    sims = (A @ q.T).ravel()
    k = min(top_k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# --------------------- VOCABS & FUZZY HELPERS ---------------------
@st.cache_data
def build_vocab(df: pd.DataFrame):
    def flat_unique(series):
        out, seen = [], set()
        for row in series:
            for x in (row or []):
                s = str(x).strip()
                if not s: continue
                k = s.lower()
                if k not in seen:
                    seen.add(k); out.append(s)
        return sorted(out)
    return (
        flat_unique(df["employers_list"]),
        flat_unique(df["colleges_list"]),
        flat_unique(df["skills_list"]),
        flat_unique(df["interests_list"]),
        flat_unique(df["locations_list"]),
    )

def try_import_rapidfuzz():
    try:
        from rapidfuzz import process, fuzz
        return process, fuzz
    except Exception:
        return None, None

PROCESS, FUZZ = try_import_rapidfuzz()
STOP_TOKENS = {"at","in","on","of","for","with","from","by","to","some","point","any","ever","the","a","an","that"}

def trim_tail(s: str) -> str:
    toks = re.findall(r"[A-Za-z&.\-]+", s)
    keep = []
    for t in toks:
        if t.lower() in STOP_TOKENS: break
        keep.append(t)
        if len(keep) >= 4: break
    return " ".join(keep)

def fuzzy_to_vocab(guess: str, vocab: list[str], cutoff: int = 80) -> str | None:
    if not guess or not vocab:
        return None
    if PROCESS and FUZZ:
        vocab_lower = [v.lower() for v in vocab]
        m = PROCESS.extractOne(guess.lower(), vocab_lower, scorer=FUZZ.WRatio, score_cutoff=cutoff)
        if m:
            return vocab[vocab_lower.index(m[0])]
    # fallback: exact or startswith
    gl = guess.lower()
    for v in vocab:
        if v.lower() == gl: return v
    for v in vocab:
        if v.lower().startswith(gl): return v
    return None

def company_from_question(q: str, emp_vocab: list[str]) -> str | None:
    m = re.search(r"(?:worked at|at|from|with)\s+([A-Za-z&.\- ]{2,60})", q, re.I)
    guess = trim_tail(m.group(1)) if m else q
    return fuzzy_to_vocab(guess, emp_vocab, cutoff=80)

def college_from_question(q: str, col_vocab: list[str]) -> str | None:
    m = re.search(r"(?:from|at)\s+([A-Za-z&.\- ]{2,60})\s+(?:university|college|institute|school)", q, re.I)
    if not m: return None
    guess = trim_tail(m.group(1))
    return fuzzy_to_vocab(guess, col_vocab, cutoff=75)

# ----- Stricter skill detection (only when question indicates skills/tools) -----
SKILL_TRIGGERS = re.compile(
    r"\b(skill|skills|know|proficient|experience\s+with|using|tool|language)\b",
    re.I,
)
def simple_skill_from_question(q: str, sk_vocab: list[str]) -> str | None:
    # Only consider skills if wording implies skills/tools
    if not SKILL_TRIGGERS.search(q or ""):
        return None
    ql = q.lower()
    # exact word-boundary hit from vocab first
    for sk in sk_vocab:
        pat = re.compile(rf"\b{re.escape(sk.lower())}\b")
        if pat.search(ql):
            return sk
    # conservative fuzzy fallback
    words = re.findall(r"[A-Za-z0-9+#.\-]{2,20}", ql)
    cand = None
    if PROCESS and FUZZ:
        best_score = 0
        vocab_lower = [s.lower() for s in sk_vocab]
        for w in words:
            m = PROCESS.extractOne(w, vocab_lower, scorer=FUZZ.WRatio, score_cutoff=93)
            if m and m[1] > best_score:
                best_score = m[1]
                cand = sk_vocab[vocab_lower.index(m[0])]
    return cand

def location_from_question(q: str, loc_vocab: list[str]) -> str | None:
    m = re.search(r"(?:in|located in|from)\s+([A-Za-z .'\-]{2,60})", q, re.I)
    if not m: return None
    guess = trim_tail(m.group(1))
    return fuzzy_to_vocab(guess, loc_vocab, cutoff=85)

def wb_contains(series: pd.Series, term: str):
    pat = re.compile(rf"\b{re.escape(term)}\b", re.I)
    return series.apply(lambda s: bool(pat.search(s or "")))

# ----- Experience type detection from question -----
EXP_TYPE_ALIASES = {
    "product management": "Product Management",
    "consulting": "Consulting",
    "consult": "Consulting",
    "education": "Education",
    "teaching": "Education",
    "finance": "Finance",
    "financial": "Finance",
    "technology": "Technology",
    "tech": "Technology",
    "marketing": "Marketing",
    "ops": "Operations",
    "operations": "Operations",
    "healthcare": "Healthcare",
    "military": "Military",
    "pm": "Product Management",
}
def exp_type_from_question(q: str) -> str | None:
    ql = (q or "").lower()
    for key in sorted(EXP_TYPE_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", ql):
            return EXP_TYPE_ALIASES[key]
    return None

# --- Experience-type keyword heuristics for soft matching ---
EXP_TYPE_KEYWORDS = {
    "Consulting": [
        r"\bconsult", r"\bcase team", r"\bstrategy & analytics",
        r"\bmckinsey", r"\bbcg\b", r"\bbain\b", r"\boliver\swyman",
        r"\bstrategy\s*&?\s*", r"\bmonitor deloitte", r"\baccenture\b",
        r"\bdeloitte\b", r"\bey\b", r"\bpwc\b", r"\bkpmg\b", r"\bparthenon\b",
    ],
    "Finance": [
        r"\binvestment bank", r"\bprivate equity", r"\bventure capital", r"\bheg[e]?\bfund",
        r"\basset management", r"\btrading\b", r"\bm&a\b", r"\bmergers?\s*&\s*acquisitions",
        r"\bjp\s*morgan", r"\bmorgan\s*stanley", r"\bgoldman", r"\bblackrock", r"\bblackstone",
    ],
    "Technology": [
        r"\bsoftware", r"\bengineer", r"\bdata\b", r"\bml\b", r"\bmachine learning",
        r"\bcloud\b", r"\bpython\b", r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bproduct\s*tech",
    ],
    "Product Management": [
        r"\bproduct manager", r"\bpm\b", r"\broadmap", r"\bbacklog", r"\bagile\b",
    ],
    "Healthcare": [r"\bhealthcare", r"\bhospital\b", r"\bclinical\b", r"\bmedical\b", r"\bpharma\b", r"\bbiotech\b"],
    "Marketing":  [r"\bmarketing\b", r"\bbrand\b", r"\bcampaign\b", r"\badvertis(ing|ement)"],
    "Operations": [r"\boperations\b", r"\bsupply chain", r"\blogistics\b", r"\bmanufacturing"],
    "Military":   [r"\barmy\b", r"\bnavy\b", r"\bair force\b", r"\bmarine(s)?\b", r"\bofficer\b", r"\bcommander\b"],
    "Education":  [r"\bteacher\b", r"\bteaching\b", r"\beducation\b", r"\bschool\b", r"\bcurriculum\b"],
}
def build_exp_mask(df_subset: pd.DataFrame, exp_label: str) -> pd.Series:
    """Soft OR mask using experience_type + role/employer/raw keywords."""
    exp_label = exp_label.strip()
    mask = pd.Series(False, index=df_subset.index)

    # 1) direct experience_type string match
    mask |= df_subset["experience_type"].str.contains(exp_label, case=False, na=False)

    # 2) keywords across roles/employers/raw text
    patterns = EXP_TYPE_KEYWORDS.get(exp_label, [])
    if patterns:
        combined = df_subset["raw_lower"] + " | " + df_subset["employers_blob"] + " | " + df_subset["colleges_blob"]
        for pat in patterns:
            rx = re.compile(pat, re.I)
            mask |= combined.apply(lambda s: bool(rx.search(s or "")))

    return mask

# --------------------- GROUNDED COUNTS ---------------------
@st.cache_data
def compute_facets(df: pd.DataFrame):
    consult_kw = ("consult", "strategy & analytics", "case team")
    flags = []
    for _, r in df.iterrows():
        et = (r.get("experience_type") or "").lower()
        roles = " ".join(r.get("roles_extracted") or []).lower()
        flags.append(("consulting" in et) or any(k in roles for k in consult_kw))
    return {"total_consult": int(sum(flags)), "n": int(len(df))}

def grounded_facts(question: str, df_full: pd.DataFrame, facets: dict,
                   emp_vocab, col_vocab, sk_vocab, loc_vocab, implicit_exp: str | None) -> str:
    facts = []
    ql = question.lower()
    if "consult" in ql:
        facts.append(f"DATA: People with consulting experience = {facets['total_consult']} of {facets['n']}.")

    comp = company_from_question(question, emp_vocab)
    if comp:
        pat = re.compile(rf"\b{re.escape(comp)}\b", re.I)
        m = (df_full["employers_blob"].apply(lambda s: bool(pat.search(s))) |
             df_full["raw_lower"].apply(lambda s: bool(pat.search(s))))
        facts.append(f"DATA: People mentioning '{comp}' (employers/text) = {int(m.sum())}.")

    col = college_from_question(question, col_vocab)
    if col:
        pat = re.compile(rf"\b{re.escape(col)}\b", re.I)
        m = (df_full["colleges_blob"].apply(lambda s: bool(pat.search(s))) |
             df_full["raw_lower"].apply(lambda s: bool(pat.search(s))))
        facts.append(f"DATA: People mentioning college '{col}' = {int(m.sum())}.")

    sk = simple_skill_from_question(question, sk_vocab)
    if sk:
        pat = re.compile(rf"\b{re.escape(sk)}\b", re.I)
        m = df_full["skills_blob"].apply(lambda s: bool(pat.search(s)))
        facts.append(f"DATA: People listing skill '{sk}' = {int(m.sum())}.")

    loc = location_from_question(question, loc_vocab)
    if loc:
        pat = re.compile(rf"\b{re.escape(loc)}\b", re.I)
        m = (df_full["locations_blob"].apply(lambda s: bool(pat.search(s))) |
             df_full["raw_lower"].apply(lambda s: bool(pat.search(s))))
        facts.append(f"DATA: People mentioning location '{loc}' = {int(m.sum())}.")

    if implicit_exp:
        m = build_exp_mask(df_full, implicit_exp)
        facts.append(f"DATA: People with experience type '{implicit_exp}' (soft match) = {int(m.sum())}.")

    return "\n".join(facts)

# --------------------- GENERATION ---------------------
def call_gemma(snippets: list[str], question: str, facts: str, timeout_s: float):
    # Compatible with older Ollama CLIs (no -o flags)
    system = (
        "You are a careful analyst. Answer using ONLY the provided snippets and DATA lines. "
        "Use the DATA counts for numeric answers. If insufficient evidence, say you don't know."
    )
    context = "\n\n".join([f"SNIPPET {i+1}:\n{snip}" for i, snip in enumerate(snippets)])
    prompt = f"{system}\n\n{facts}\n\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    code, out, err = run_cmd(["ollama", "run", GEN_MODEL], stdin=prompt, timeout=timeout_s)
    if code != 0:
        return f"[LLM error] {err or out}".strip()
    return out.strip()

# --------------------- UI ---------------------
df = load_enriched(ENRICHED_JSONL)
backend, E, meta = build_or_load_index(df)
facets = compute_facets(df)
EMP_VOCAB, COL_VOCAB, SK_VOCAB, INT_VOCAB, LOC_VOCAB = build_vocab(df)

with st.sidebar:
    st.header("Filters")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    sel_companies = st.multiselect("Employers", EMP_VOCAB)
    sel_colleges  = st.multiselect("Colleges",  COL_VOCAB)
    sel_skills    = st.multiselect("Skills",    SK_VOCAB)
    sel_interests = st.multiselect("Interests", INT_VOCAB)
    sel_locations = st.multiselect("Locations", LOC_VOCAB)

    st.header("Retrieval")
    top_k     = st.slider("Top K documents", 3, 30, 10, 1)
    timeout_s = st.slider("Per-call timeout (sec)", 3, 30, 10, 1)
    st.metric("Embedding backend", backend.upper())
    if backend == "py":
        st.caption("Tip: pip install sentence-transformers")
    if PROCESS is None:
        st.caption("Tip: pip install rapidfuzz (better fuzzy matching)")

ensure_gen_model()

q  = st.text_input("Ask a question (semantic):", placeholder="e.g., How many people worked in Consulting?")
go = st.button("Search")

if go and q.strip():
    # Auto-derive tokens from the question
    implicit_comp = company_from_question(q, EMP_VOCAB)
    implicit_col  = college_from_question(q, COL_VOCAB)
    implicit_sk   = simple_skill_from_question(q, SK_VOCAB)
    implicit_loc  = location_from_question(q, LOC_VOCAB)
    implicit_exp  = exp_type_from_question(q)

    msgs = []
    if implicit_comp: msgs.append(f"🔎 Interpreted company: **{implicit_comp}**")
    if implicit_col:  msgs.append(f"🔎 Interpreted college: **{implicit_col}**")
    if implicit_sk:   msgs.append(f"🔎 Interpreted skill: **{implicit_sk}**")
    if implicit_loc:  msgs.append(f"🔎 Interpreted location: **{implicit_loc}**")
    if implicit_exp:  msgs.append(f"🔎 Interpreted experience type: **{implicit_exp}**")
    if msgs: st.caption(" • ".join(msgs))

    # Candidate set via facet filters (explicit + implicit)
    candidate = df.copy()
    filter_msgs = []

    def union_mask(curr: pd.DataFrame, terms: list[str], blob_col: str, also_raw=True):
        if not terms:
            return pd.Series(True, index=curr.index)
        mask = pd.Series(False, index=curr.index)
        for t in terms:
            mask |= wb_contains(curr[blob_col], t)
            if also_raw:
                mask |= wb_contains(curr["raw_lower"], t)
        return mask

    # Employers (OR within facet)
    emp_terms = list(sel_companies)
    if implicit_comp and implicit_comp not in emp_terms:
        emp_terms.append(implicit_comp)
    if emp_terms:
        m = union_mask(candidate, emp_terms, "employers_blob", also_raw=True)
        filter_msgs.append(f"Employer filter: {', '.join(emp_terms)} → {int(m.sum())} matches")
        candidate = candidate[m]

    # Colleges (OR)
    col_terms = list(sel_colleges)
    if implicit_col and implicit_col not in col_terms:
        col_terms.append(implicit_col)
    if col_terms:
        m = union_mask(candidate, col_terms, "colleges_blob", also_raw=True)
        filter_msgs.append(f"College filter: {', '.join(col_terms)} → {int(m.sum())} matches")
        candidate = candidate[m]

    # Skills (OR)
    sk_terms = list(sel_skills)
    if implicit_sk and implicit_sk not in sk_terms:
        sk_terms.append(implicit_sk)
    if sk_terms:
        m = union_mask(candidate, sk_terms, "skills_blob", also_raw=False)
        filter_msgs.append(f"Skills filter: {', '.join(sk_terms)} → {int(m.sum())} matches")
        candidate = candidate[m]

    # Interests (OR)
    if sel_interests:
        m = union_mask(candidate, list(sel_interests), "interests_blob", also_raw=False)
        filter_msgs.append(f"Interests filter: {', '.join(sel_interests)} → {int(m.sum())} matches")
        candidate = candidate[m]

    # Locations (OR)
    loc_terms = list(sel_locations)
    if implicit_loc and implicit_loc not in loc_terms:
        loc_terms.append(implicit_loc)
    if loc_terms:
        m = union_mask(candidate, loc_terms, "locations_blob", also_raw=True)
        filter_msgs.append(f"Location filter: {', '.join(loc_terms)} → {int(m.sum())} matches")
        candidate = candidate[m]

    # Experience Type (soft OR mask across fields). Never block the query.
    if implicit_exp:
        m_exp = build_exp_mask(candidate, implicit_exp)
        count_exp = int(m_exp.sum())
        st.caption(f"Experience type filter (soft): {implicit_exp} → {count_exp} matches")
        if count_exp > 0:
            candidate = candidate[m_exp]

    if filter_msgs:
        st.caption(" • ".join(filter_msgs))

    # If ALL filters wiped the set, drop ONLY the exp-type filter and continue
    if candidate.empty and implicit_exp:
        fallback = df.copy()
        if emp_terms:
            m = union_mask(fallback, emp_terms, "employers_blob", also_raw=True); fallback = fallback[m]
        if col_terms:
            m = union_mask(fallback, col_terms, "colleges_blob", also_raw=True);  fallback = fallback[m]
        if sk_terms:
            m = union_mask(fallback, sk_terms, "skills_blob",   also_raw=False);   fallback = fallback[m]
        if sel_interests:
            m = union_mask(fallback, list(sel_interests), "interests_blob", also_raw=False); fallback = fallback[m]
        if loc_terms:
            m = union_mask(fallback, loc_terms, "locations_blob", also_raw=True); fallback = fallback[m]
        if not fallback.empty:
            candidate = fallback
            st.caption("⚠️ No matches with experience-type filter; fell back to other filters so Gemma can still answer.")

    if candidate.empty:
        st.warning("No resumes match the current filters. Clear some filters and try again.")
        st.stop()

    # 1) Embed query
    try:
        q_emb = embed_query(q, backend)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()

    # 2) Retrieve within candidate set
    if len(candidate) == len(df):
        E_sub = E
        base_idx = df.index.to_list()
    else:
        base_idx = candidate.index.to_list()
        E_sub = E[base_idx]

    idx, sims = cosine_search(E_sub, q_emb, top_k=top_k)
    retrieved = candidate.iloc[idx].copy()
    snippets  = retrieved["chunk"].tolist()

    # 3) Grounded facts (global counts over full df)
    facts = grounded_facts(q, df, facets, EMP_VOCAB, COL_VOCAB, SK_VOCAB, LOC_VOCAB, implicit_exp)

    # 4) Ask Gemma
    with st.spinner("Asking Gemma…"):
        answer = call_gemma(snippets, q, facts, timeout_s=timeout_s)

    st.subheader("Answer")
    st.write(answer)

    # 5) Sources
    st.subheader("Sources")
    src_df = pd.DataFrame({
        "Similarity": [round(float(s), 3) for s in sims],
        "Name": retrieved["name"].tolist(),
        "Experience Type": retrieved["experience_type"].tolist(),
        "Employers": [", ".join(x or []) for x in retrieved["employers_list"]],
        "Colleges":  [", ".join(x or []) for x in retrieved["colleges_list"]],
        "Skills":    [", ".join(x or []) for x in retrieved["skills_list"]],
        "Interests": [", ".join(x or []) for x in retrieved["interests_list"]],
        "Locations": [", ".join(x or []) for x in retrieved["locations_list"]],
    })
    st.dataframe(src_df, use_container_width=True, height=360)

    with st.expander("Retrieved snippets"):
        for i, sn in enumerate(snippets, 1):
            st.code(sn, language="markdown")
else:
    st.info("Use facet filters (Employers, Colleges, Skills, Interests, Locations) and ask a question. The app pre-filters by facets, semantically re-ranks matches, and uses DATA counts to ground numeric answers.")
