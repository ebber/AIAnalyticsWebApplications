#!/usr/bin/env python3
# make_summaries.py
import argparse, json, re
from pathlib import Path

def to_list_norm(x):
    """Normalize LLM outputs: allow string with commas/semicolons or list."""
    if not x: return []
    if isinstance(x, list):
        items = x
    else:
        items = re.split(r'[;,]', str(x))
    norm, seen = [], set()
    for it in items:
        s = it.strip()
        if not s: 
            continue
        k = s.lower()
        if k not in seen:
            seen.add(k); norm.append(s)
    return norm

def first_nonempty(*args):
    for a in args:
        if isinstance(a, list) and a:
            return a
    return []

def main():
    ap = argparse.ArgumentParser(description="Emit 5-field summaries JSON (+Name)")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    summaries = []
    with Path(args.in_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            skills = first_nonempty(
                to_list_norm(rec.get("skills_llm", [])),
                to_list_norm(rec.get("skills_rule", []))
            )
            interests = first_nonempty(
                to_list_norm(rec.get("interests_llm", [])),
                to_list_norm(rec.get("interests_rule", []))
            )

            summaries.append({
                "Name": rec.get("name_extracted", ""),  # <-- added
                "Undergrad Institution(s)": rec.get("undergrad_extracted", []),
                "Location(s)": rec.get("locations_extracted", []),
                "Experience Type": rec.get("experience_type", "General Management"),
                "Skills": skills[:15],
                "Interests": interests[:10],
            })

    Path(args.out_json).write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote summaries → {args.out_json}")

if __name__ == "__main__":
    main()
