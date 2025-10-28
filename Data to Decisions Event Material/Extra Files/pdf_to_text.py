#!/usr/bin/env python3
# pdf_to_text.py
import argparse, hashlib, json, re
from pathlib import Path
import fitz  # PyMuPDF

SECTION_HEADERS = {
    "EDUCATION","EXPERIENCE","ADDITIONAL DATA","ADDITIONAL INFORMATION",
    "LEADERSHIP","PROJECTS","SKILLS","CERTIFICATIONS"
}

def normalize(s: str) -> str:
    s = s.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    s = re.sub(r'[–—]', '-', s)
    s = re.sub(r'[•·●▪■□]', '•', s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s

def page_to_lines(page):
    d = page.get_text("dict")
    lines = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text","") for span in line.get("spans",[])).strip()
            if text:
                lines.append(text)
    if not lines:
        lines = [t.strip() for t in page.get_text().splitlines() if t.strip()]
    return [normalize(l) for l in lines]

def is_name_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 4 or len(line) > 60: return False
    if not re.match(r"^[A-Za-z][A-Za-z\s.\-']+$", line): return False
    L = line.lower()
    if any(x in L for x in ['@','|','http','www','(']): return False
    if any(x in line.upper() for x in ['UNIVERSITY','COLLEGE','SCHOOL','INSTITUTE']): return False
    words = line.split()
    if not (2 <= len(words) <= 5): return False
    capsish = sum(1 for w in words if re.match(r'^[A-Z][a-zA-Z\-\.]*$', w))
    return capsish >= 2

def confirm_header(lines: list[str], idx: int) -> bool:
    n = len(lines)
    contact_win = "\n".join(lines[idx:min(idx+6, n)])
    has_contact = re.search(r'@|(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', contact_win) is not None
    edu_win = "\n".join(lines[idx:min(idx+15, n)]).upper()
    has_edu = "EDUCATION" in edu_win
    return has_contact and has_edu

def split_resumes(all_lines: list[str]) -> list[list[str]]:
    """
    Start a resume ONLY on a valid name line that passes confirm_header.
    Accumulate until the next confirmed name header. Never start on section headers.
    """
    chunks: list[list[str]] = []
    i, n = 0, len(all_lines)
    current: list[str] | None = None

    while i < n:
        line = all_lines[i].strip()

        if current is None:
            if is_name_line(line) and confirm_header(all_lines, i):
                current = [line]
            i += 1
            continue

        # We have a current resume
        if is_name_line(line) and confirm_header(all_lines, i):
            if len(current) >= 8:
                chunks.append(current)
            current = [line]
            i += 1
            continue

        current.append(line)
        i += 1

    if current and len(current) >= 8:
        chunks.append(current)
    return chunks

def is_orphan_section(lines: list[str]) -> bool:
    if not lines: return False
    first = lines[0].strip().upper()
    if first in SECTION_HEADERS:
        return True
    blob = "\n".join(lines)
    has_contact = re.search(r'@|(?:\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', blob) is not None
    has_edu = "EDUCATION" in blob.upper()
    return (not has_contact) and (not has_edu)

def stitch_orphans(chunks: list[list[str]]) -> list[list[str]]:
    if not chunks: return chunks
    stitched: list[list[str]] = []
    for ch in chunks:
        if stitched and is_orphan_section(ch):
            stitched[-1].extend(ch)
        else:
            stitched.append(ch)
    return stitched

def main():
    ap = argparse.ArgumentParser(description="PDF → per-resume raw text (JSONL)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL, one resume per line")
    ap.add_argument("--limit", type=int, help="Optional: only first N resumes")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    all_lines = []
    for p in doc:
        all_lines.extend(page_to_lines(p))
    doc.close()

    chunks = split_resumes(all_lines)
    chunks = stitch_orphans(chunks)
    if args.limit:
        chunks = chunks[:args.limit]

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for ch in chunks:
            text = "\n".join(ch)
            rid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            f.write(json.dumps({"resume_id": rid, "raw_text": text}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(chunks)} resumes → {out}")

if __name__ == "__main__":
    main()
