#!/usr/bin/env python3
# extract_fields.py
import argparse, json, re
from pathlib import Path

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
CITY_STATE = re.compile(r'\b([A-Za-z][A-Za-z.\'\- ]{1,40}),\s*([A-Z]{2})\b')
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.\w+\b', re.IGNORECASE)
PHONE = re.compile(r'\+?\d{0,2}\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}')
DATE_LINE = re.compile(r'(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?[a-z]*\s+\d{4}\s*[-–—]\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?[a-z]*\s+\d{4})')
ROLE_TITLE = re.compile(r'(?i)\b(Chief|Senior|Lead|Head of|VP|Vice President|Director|Manager|Associate|Analyst|Consultant|Engineer|Specialist|Coordinator|Officer|Teacher|Instructor|Educator|Coach|Founder|Owner|Partner|Intern|Assistant)\b')
ROLE_FUZZY = ("teacher","instructor","educator","coach","professor","tutor","dean","principal")

SECTION_HEADERS = {
    "EDUCATION","EXPERIENCE","ADDITIONAL DATA","ADDITIONAL INFORMATION",
    "LEADERSHIP","PROJECTS","SKILLS","CERTIFICATIONS"
}

UNDERGRAD_BLACKLIST = ("HONORS","THESIS","MAJOR","MINOR","GPA","COURSE","COURSES","DEAN",
                       "CANDIDATE","FOCUS AREAS","EXPERIENTIAL","EXTRACURRICULAR","AWARDS","INTERNSHIP","INTERNSHIPS")

SKILL_HINTS = ("skill","skills","technical","proficiency","language","tools")
INTEREST_HINTS = ("interest","interests","hobby","hobbies","activities")

def dedupe(seq):
    seen, out = set(), []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k); out.append(s.strip())
    return out

def slice_section(raw_text: str, header: str) -> list[str]:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    out, in_sec = [], False
    for l in lines:
        u = l.upper().strip()
        if u == header:
            in_sec = True
            continue
        if in_sec and u in SECTION_HEADERS:
            break
        if in_sec:
            out.append(l)
    return out

def get_header_lines(raw_text: str, k: int = 6) -> list[str]:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    return lines[:k]

def extract_name(raw_text: str) -> str:
    lines = get_header_lines(raw_text, 4)
    for l in lines:
        if is_name_line(l):
            return l.title().strip()
    return ""

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

def extract_locations_scoped(raw_text: str):
    """Only scan header + EXPERIENCE (skip colon lines to avoid 'PowerBI, MS')."""
    locs = []
    # header
    for l in get_header_lines(raw_text, 6):
        for m in CITY_STATE.finditer(l):
            city, st = m.group(1).strip(), m.group(2)
            if st in US_STATES:
                locs.append(f"{city}, {st}")
    # experience section
    for l in slice_section(raw_text, "EXPERIENCE"):
        if ":" in l:  # heuristically skip descriptor lines that cause false matches
            continue
        for m in CITY_STATE.finditer(l):
            city, st = m.group(1).strip(), m.group(2)
            if st in US_STATES:
                locs.append(f"{city}, {st}")
    return dedupe(locs)

def extract_contacts(text):
    email = EMAIL.search(text)
    phone = PHONE.search(text)
    return (email.group(0).lower() if email else ""), (phone.group(0) if phone else "")

def extract_undergrad_from_education(raw_text):
    edu_lines = slice_section(raw_text, "EDUCATION")
    out = []
    for l in edu_lines:
        U = l.upper().strip()
        if any(skip in U for skip in ("KELLOGG","NORTHWESTERN")):
            # skip grad program lines
            continue
        if any(b in U for b in UNDERGRAD_BLACKLIST):
            continue
        if any(kw in U for kw in ("UNIVERSITY","COLLEGE","INSTITUTE","SCHOOL OF")) and len(l) > 8:
            clean = re.sub(r',\s*[A-Z]{2}$', '', l.strip())
            out.append(clean)
    return dedupe(out)[:3]

def extract_roles(text):
    roles = []
    for line in text.splitlines():
        s = line.strip()
        if not s or '•' in s: 
            continue
        if DATE_LINE.search(s) and ROLE_TITLE.search(s):
            pre = DATE_LINE.split(s, maxsplit=1)[0].strip(" ,-|")
            if pre:
                roles.append(pre)
        elif ROLE_TITLE.search(s) or any(tok in s.lower() for tok in ROLE_FUZZY):
            roles.append(s)
    return dedupe(roles)[:12]

def extract_additional_rules(raw_text):
    """Rule-based skills/interests from ADDITIONAL sections."""
    skills, interests = [], []
    add_lines = slice_section(raw_text, "ADDITIONAL DATA") + slice_section(raw_text, "ADDITIONAL INFORMATION") + slice_section(raw_text, "SKILLS")
    for l in add_lines:
        low = l.lower()
        if any(h in low for h in SKILL_HINTS):
            content = l.split(":", 1)[-1] if ":" in l else l
            parts = re.split(r'[;,|•]', content)
            for p in parts:
                p = p.strip()
                if 2 < len(p) <= 50:
                    skills.append(p)
        elif any(h in low for h in INTEREST_HINTS):
            content = l.split(":", 1)[-1] if ":" in l else l
            parts = re.split(r'[;,|•]', content)
            for p in parts:
                p = p.strip()
                if 2 < len(p) <= 80:
                    interests.append(p)
    return dedupe(skills)[:20], dedupe(interests)[:12]

def looks_like_section_only(raw_text: str) -> bool:
    first = raw_text.splitlines()[0].strip().upper() if raw_text.strip() else ""
    return first in SECTION_HEADERS

def main():
    ap = argparse.ArgumentParser(description="Deterministic field extraction")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--skip_orphans", action="store_true", help="Drop section-only records")
    args = ap.parse_args()

    out = Path(args.out_jsonl).open("w", encoding="utf-8")
    count_in = count_out = 0
    with Path(args.in_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            count_in += 1
            rec = json.loads(line)
            raw = rec["raw_text"]

            if args.skip_orphans and looks_like_section_only(raw):
                continue

            name = extract_name(raw)
            locs = extract_locations_scoped(raw)
            email, phone = extract_contacts(raw)
            roles = extract_roles(raw)
            undergrad = extract_undergrad_from_education(raw)
            skills_rule, interests_rule = extract_additional_rules(raw)

            rec.update({
                "name_extracted": name,
                "locations_extracted": locs,
                "email_extracted": email,
                "phone_extracted": phone,
                "roles_extracted": roles,
                "undergrad_extracted": undergrad,
                "skills_rule": skills_rule,
                "interests_rule": interests_rule
            })
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count_out += 1
    out.close()
    print(f"[OK] processed {count_in} → wrote {count_out} to {args.out_jsonl}")

if __name__ == "__main__":
    main()
