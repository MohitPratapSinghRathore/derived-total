"""Playbook section 8 QA gates. Every one must pass before the paper is 'done'.

Run after building: python code/qa_gates.py
Exits non-zero if any gate fails, so it can sit in CI.
"""
import os, re, sys, subprocess, json

ROOT = os.path.join(os.path.dirname(__file__), "..")
PAPER = os.path.join(ROOT, "paper")
BUILD = os.path.join(ROOT, "build")

NAMES = ["Rathore", "Gunveer", "Kalsi", "Meduri", "Mohit", "Sriharsha"]
AFFILS = ["Oviqo"]
EMAIL = "gunveerkalsi@gmail.com"

fails, warns = [], []


def gate(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def pdf_text(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


def main():
    print("QA gates")

    # ---- source-level gates (catch problems the PDF may hide)
    src = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    body = src
    gate("---" not in body, "no em-dash markup in source (---)")
    gate(not re.search(r"(?<![\d])--(?![\d-])", body),
         "no bare -- used as punctuation in source")
    gate("robust" not in body.lower(), "no 'robust' in source")

    # title <= 12 words
    m = re.search(r"\\title\{\\bf (.+?)\}\s*\n", src, re.S)
    if m:
        t = re.sub(r"\\\\|\{|\}", " ", m.group(1))
        nw = len(t.split())
        gate(nw <= 12, "title <= 12 words", f"{nw} words")

    # abstract <= 250 words
    a = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", src, re.S)
    if a:
        txt = re.sub(r"\\result\{[^}]*\}", "X", a.group(1))
        txt = re.sub(r"\\[a-zA-Z]+\*?|\{|\}|\$", " ", txt)
        nw = len(txt.split())
        gate(nw <= 250, "abstract <= 250 words", f"{nw} words")

    # Every measured number in prose must come from \result{}. The check covers
    # the body only: the methods appendix states configuration constants such as
    # weight decay and a feed-forward multiplier, which are inputs to the
    # experiment rather than outputs of it.
    body_src = src.split("\\appendix")[0]
    prose = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body_src, flags=re.S)
    prose = re.sub(r"\\input\{[^}]*\}", "", prose)
    prose = re.sub(r"\$[^$]*\$", "", prose)          # maths is not a result claim
    # layout lengths are not result claims either
    prose = re.sub(r"\\includegraphics\[[^]]*\]", "", prose)
    prose = re.sub(r"\\(?:v|h)space\{[^}]*\}", "", prose)
    prose = re.sub(r"\[width=[^]]*\]", "", prose)
    bare = re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", prose)
    gate(len(bare) == 0, "no hardcoded decimal figures in prose",
         f"found {bare[:5]}" if bare else "")

    # ---- staleness. A failing build once went unnoticed for a day because the
    # output directory still held an older PDF and nothing checked its age.
    pdf = os.path.join(BUILD, "main.pdf")
    if os.path.exists(pdf):
        newest_src = max(os.path.getmtime(os.path.join(PAPER, f))
                         for f in ("main.tex", "tables.tex", "results_macros.tex")
                         if os.path.exists(os.path.join(PAPER, f)))
        age = os.path.getmtime(pdf) - newest_src
        gate(age > 0, "built PDF is newer than its sources",
             f"PDF is {abs(age)/3600:.1f} h older than the newest source"
             if age <= 0 else "")

    # ---- built-artifact gates
    t = pdf_text(pdf)
    if t is None:
        warns.append("main.pdf not built or pypdf missing; PDF gates skipped")
    else:
        gate(t.count("\u2014") == 0, "0 em-dashes in PDF", f"{t.count(chr(8212))}")
        gate(t.lower().count("robust") == 0, "0 'robust' in PDF")
        gate(t.count("??") == 0, "0 undefined refs/cites/results in PDF",
             f"{t.count('??')} occurrences")
        # body word count
        i = t.find("Introduction")
        wc = len(t[i:].split()) if i > 0 else len(t.split())
        print(f"  [INFO] body word count ~{wc}")

    anon = pdf_text(os.path.join(BUILD, "main_anon.pdf"))
    if anon is None:
        warns.append("main_anon.pdf not built; anonymity gate skipped")
    else:
        leaks = [k for k in NAMES + AFFILS + [EMAIL] if k.lower() in anon.lower()]
        gate(not leaks, "anonymous build has 0 identity leaks", str(leaks))

    # ---- results provenance
    prov = os.path.join(ROOT, "results", "macro_provenance.json")
    if os.path.exists(prov):
        d = json.load(open(prov))
        na = [k for k, v in d.items() if v["value"] == "n/a"]
        print(f"  [INFO] {len(d)} result macros, {len(na)} still n/a")
        used = set(re.findall(r"\\result\{([^}]*)\}", src))
        missing = sorted(used - set(d))
        gate(not missing, "every \\result key used in the paper is defined",
             str(missing[:8]))
    else:
        warns.append("macro_provenance.json missing; run code/make_macros.py")

    # ---- no --- in generated tables
    tbl = os.path.join(PAPER, "tables.tex")
    if os.path.exists(tbl):
        gate("---" not in open(tbl, encoding="utf-8").read(),
             "no --- in generated tables (use n/a)")

    print()
    for w in warns:
        print(f"  [WARN] {w}")
    if fails:
        print(f"\n{len(fails)} gate(s) FAILED: {fails}")
        sys.exit(1)
    print("\nAll runnable gates passed.")


if __name__ == "__main__":
    main()
