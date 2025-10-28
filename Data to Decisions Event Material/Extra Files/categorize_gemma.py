#!/usr/bin/env python3
# categorize_gemma.py  (robust: labeled text output -> regex parse)
import argparse, json, orjson, subprocess, sys, time, re
from pathlib import Path
from subprocess import TimeoutExpired
from collections import Counter

DEFAULT_MODEL = "gemma3:1b"   # use your tag here

CATEGORIES = [
    "Consulting","Finance","Technology","Product Management","Healthcare",
    "Marketing","Operations","Military","Education","Customer Success",
    "Nonprofit/Public Service","General Management"
]

# Lightweight rule keywords (fast path)
RULE_KW = {
    "Consulting": ["consult", "case team", "engagement", "strategy consulting", "deloitte", "mckinsey", "bcg", "bain"],
    "Finance": ["investment bank", "private equity", "m&a", "leveraged buyout", "goldman", "morgan stanley", "jp morgan", "hedge fund", "valuation", "buyout"],
    "Technology": ["software", "developer", "engineer", "data", "python", "java", "cloud", "aws", "azure", "kubernetes", "ml", "ai"],
    "Product Management": ["product manager", "product owner", "roadmap", "backlog", "prioritization", "user stories", "pm"],
    "Healthcare": ["healthcare", "hospital", "clinical", "pharma", "biotech", "med device", "payer", "provider"],
    "Marketing": ["brand", "campaign", "go-to-market", "segmentation", "marketing", "growth marketing"],
    "Operations": ["operations", "supply chain", "logistics", "manufacturing", "ops"],
    "Military": ["army", "navy", "air force", "marine", "battalion", "brigade", "commander"],
    "Education": ["teacher", "teaching", "education", "curriculum", "school", "edtech", "classroom", "i-ready"],
    "Customer Success": ["customer success", "partner success", "renewals", "adoption", "retention", "onboarding", "enablement"],
    "Nonprofit/Public Service": ["nonprofit", "ngo", "public sector", "government", "americorps", "fellowship"],
}

SYSTEM_PROMPT = f"""You are a precise resume classifier.
Pick 1–2 categories from this set (use Title Case, separated by ' / ' if two):
{", ".join(CATEGORIES)}

You MUST output EXACTLY THREE lines, nothing else:
EXPERIENCE_TYPE=Category or "A / B"
SKILLS=skill1; skill2; ... (<=15 total; concise; canonical)
INTERESTS=interest1; interest2; ... (<=10 total; concise)

Absolutely NO prose or explanations. Only those three lines.
"""

USER_TEMPLATE = """EVIDENCE (condensed):
- Roles: {roles}
- Undergrad: {undergrad}
- Locations: {locs}
- Email: {email}; Phone: {phone}

RAW (trimmed):
{raw}
"""

EXPR_ET = re.compile(r'^\s*EXPERIENCE_TYPE\s*=\s*(.+?)\s*$', re.I | re.M)
EXPR_SK = re.compile(r'^\s*SKILLS\s*=\s*(.+?)\s*$', re.I | re.M)
EXPR_IN = re.compile(r'^\s*INTERESTS\s*=\s*(.+?)\s*$', re.I | re.M)

def _default_result():
    return {"experience_type":"General Management","skills":[],"interests":[]}

def fast_rule_label(roles, undergrad, locations, raw, min_score=3, margin=2):
    text_heavy = (" ".join(roles + undergrad + locations)).lower()
    text_light = raw.lower()
    scores = Counter()
    for lbl, kws in RULE_KW.items():
        for kw in kws:
            if kw in text_heavy:
                scores[lbl] += 3
            if kw in text_light:
                scores[lbl] += 1
    if not scores:
        return None
    ranked = scores.most_common()
    top_label, top_score = ranked[0]
    lead = top_score - (ranked[1][1] if len(ranked) > 1 else 0)
    if top_score >= min_score and lead >= margin:
        second = [l for l, s in ranked[1:2] if s >= min_score]
        if second:
            return " / ".join(sorted([top_label] + second)[:2])
        return top_label
    return None

def parse_three_lines(text: str):
    """
    Parse the 3-line constrained output.
    Returns dict or default if missing.
    """
    et = EXPR_ET.search(text)
    sk = EXPR_SK.search(text)
    inn = EXPR_IN.search(text)
    if not et:
        return _default_result()
    exp = et.group(1).strip()
    def split_items(s):
        items = [x.strip() for x in re.split(r'[;,\|]', s) if x.strip()]
        # de-dup, preserve order
        seen, out = set(), []
        for it in items:
            k = it.lower()
            if k not in seen:
                seen.add(k); out.append(it)
        return out
    skills = split_items(sk.group(1))[:15] if sk else []
    interests = split_items(inn.group(1))[:10] if inn else []
    return {"experience_type": exp, "skills": skills, "interests": interests}

def call_ollama(prompt: str, model: str, timeout_s: float, ollama_opts: list[str]):
    """
    Run `ollama run <model>` with -o options and timeout.
    Returns (parsed_dict, status_str, duration_ms, raw_text)
    """
    cmd = ["ollama", "run", model]
    for opt in ollama_opts:
        if "=" in opt:
            cmd.extend(["-o", opt])
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except TimeoutExpired:
        return _default_result(), "timeout", int((time.time() - t0) * 1000), ""
    raw_out = proc.stdout.decode("utf-8", errors="replace").strip()
    parsed = parse_three_lines(raw_out)
    if parsed == _default_result() and raw_out:
        # couldn't parse the three lines strictly
        return parsed, "parse-fail", int((time.time() - t0) * 1000), raw_out
    return parsed, "ok", int((time.time() - t0) * 1000), raw_out

def shorten(text: str, limit: int):
    if len(text) <= limit: return text
    lines = text.splitlines()
    out, total = [], 0
    for ln in lines:
        if total + len(ln) + 1 > limit: break
        out.append(ln); total += len(ln) + 1
    return "\n".join(out) if out else text[:limit]

def to_join(seq): return ", ".join(seq) if seq else ""

def main():
    ap = argparse.ArgumentParser(description="Categorize with Gemma via Ollama (regex-parsed 3-line format)")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max", type=int)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    ap.add_argument("--per_call_timeout", type=float, default=7.0)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--metrics_out", type=Path)
    ap.add_argument("--raw_limit", type=int, default=1000)
    ap.add_argument("--min_score", type=int, default=3)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--ollama_opt", action="append", default=[
        "num_predict=64",
        "temperature=0",
        "top_p=0.9",
        # optional stop token so the model stops after the three lines if it tries to ramble
        "stop=RAW (trimmed):",
    ])
    args = ap.parse_args()

    model = args.model
    timeout_s = max(2.0, args.per_call_timeout)

    out_f = Path(args.out_jsonl).open("w", encoding="utf-8")
    metrics_f = args.metrics_out.open("w", encoding="utf-8") if args.metrics_out else None

    total = ok = timeouts = parsefails = skipped = 0

    with Path(args.in_jsonl).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if args.max and i > args.max: break
            rec = json.loads(line)

            roles = rec.get("roles_extracted", [])
            undergrad = rec.get("undergrad_extracted", [])
            locs = rec.get("locations_extracted", [])
            raw = rec.get("raw_text","")

            # FAST PATH (skip LLM)
            rule_label = fast_rule_label(roles, undergrad, locs, raw, args.min_score, args.margin)
            if rule_label:
                rec["experience_type"] = rule_label
                rec["skills_llm"] = []
                rec["interests_llm"] = []
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                skipped += 1; total += 1
                if args.progress_every and (i % args.progress_every == 0):
                    print(f"[PROGRESS] {i} processed | ok={ok} timeout={timeouts} parsefail={parsefails} skipped={skipped}", file=sys.stderr)
                if metrics_f:
                    metrics_f.write(json.dumps({"i": i, "resume_id": rec.get("resume_id"), "status": "skipped-fastpath", "duration_ms": 0}) + "\n")
                continue

            # LLM PATH
            prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
                roles=to_join(roles),
                undergrad=to_join(undergrad),
                locs=to_join(locs),
                email=rec.get("email_extracted",""),
                phone=rec.get("phone_extracted",""),
                raw=shorten(raw, args.raw_limit),
            )

            llm, status, dur_ms, raw_out = call_ollama(prompt, model=model, timeout_s=timeout_s, ollama_opts=args.ollama_opt)
            total += 1
            if status == "ok":
                ok += 1
            elif status == "timeout":
                timeouts += 1
                print(f"[TIMEOUT] #{i} >{timeout_s:.1f}s → fallback", file=sys.stderr)
            elif status == "parse-fail":
                parsefails += 1
                # Log a tiny snippet to help debug formatting
                print(f"[PARSE-FAIL] #{i} Could not parse 3-line format. Snip: {raw_out[:120].replace(chr(10),' ')}...", file=sys.stderr)

            rec["experience_type"] = llm.get("experience_type", "General Management")
            rec["skills_llm"] = llm.get("skills", [])
            rec["interests_llm"] = llm.get("interests", [])
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if metrics_f:
                metrics_f.write(json.dumps({
                    "i": i,
                    "resume_id": rec.get("resume_id"),
                    "status": status,
                    "duration_ms": dur_ms
                }) + "\n")

            if args.progress_every and (i % args.progress_every == 0):
                print(f"[PROGRESS] {i} processed | ok={ok} timeout={timeouts} parsefail={parsefails} skipped={skipped}", file=sys.stderr)

    out_f.close()
    if metrics_f: metrics_f.close()
    print(f"[DONE] processed={total} | ok={ok} | timeout={timeouts} | parsefail={parsefails} | skipped-fastpath={skipped} | out={args.out_jsonl}", file=sys.stderr)

if __name__ == "__main__":
    main()
