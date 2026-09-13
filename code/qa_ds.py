"""QA gates for the differential-scaling paper, mirroring Paper 1's."""
import os
import re
import sys
import pypdf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "paper_scaling", "main.tex")
MAC = os.path.join(ROOT, "paper_scaling", "results_macros.tex")
PDF = os.path.join(ROOT, "build_scaling", "main.pdf")
fails = []


def gate(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


src = open(SRC, encoding="utf-8").read()
body = src.split(r"\begin{document}", 1)[1]
prose = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", " ", body, flags=re.S)
prose = re.sub(r"(?m)%.*$", " ", prose)
prose_nomac = re.sub(r"\\result\{[^}]*\}", " ", prose)

gate("---" not in body, "no em-dash markup")
gate(not re.search(r"\s--\s", prose_nomac), "no bare -- as punctuation")
gate("robust" not in src.lower(), "no 'robust'")

title = re.search(r"\\title\{\\bf (.*?)\}\n", src).group(1).replace("\\\\", " ")
gate(len(title.split()) <= 12, "title <= 12 words", len(title.split()))

ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
abw = len(re.sub(r"\\result\{[^}]*\}", "X", ab).split())
gate(abw <= 250, "abstract <= 250 words", abw)

scan = re.sub(r"\\(?:cite[a-z]*|ref|label|includegraphics)\{[^}]*\}", " ", prose_nomac)
dec = re.findall(r"(?<![\w.])\d+\.\d+", scan)
gate(not dec, "no hardcoded decimals in prose", dec[:6])

defined = set(re.findall(r"\\defresult\{([^}]+)\}", open(MAC, encoding="utf-8").read()))
used = set(re.findall(r"\\result\{([^}]+)\}", src))
gate(not (used - defined), "every \\result key defined", sorted(used - defined))

pdf = pypdf.PdfReader(PDF)
txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
gate("??" not in txt, "no ?? in PDF", txt.count("??"))
gate("n/a" not in open(MAC, encoding="utf-8").read().split("% AUTO")[0], "macro file sane")
newest = max(os.path.getmtime(f) for f in (SRC, MAC))
gate(os.path.getmtime(PDF) > newest, "PDF newer than sources")
print(f"  [INFO] pages {len(pdf.pages)}, macros used {len(used)} of {len(defined)}")
sys.exit(1 if fails else 0)
