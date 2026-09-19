"""QA gates for the admissibility paper, mirroring the other two."""
import os
import re
import sys
import pypdf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "paper_dsr", "main.tex")
MAC = os.path.join(ROOT, "paper_dsr", "results_macros.tex")
PDF = os.path.join(ROOT, "build_dsr", "main.pdf")
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

# A mangled \result leaves the macro name as bare prose, which compiles cleanly and
# prints "esult{key}" in the PDF. This happened once, from a shell heredoc turning
# \r into a carriage return, and passed every other gate. Catch the whole class:
# any fragment of a known control sequence appearing without its backslash.
leaked = []
for stem in ("result", "cite", "citep", "citet", "ref", "label", "textbf",
             "emph", "paragraph", "section", "item"):
    for tail in (stem[1:], stem[2:]):
        leaked += [f"{tail}{{" for _ in re.findall(r"(?<![\\\w])" + tail + r"\{", body)]
gate(not leaked, "no de-escaped control sequences in source",
     sorted(set(leaked))[:5])
gate(b"\r" not in open(SRC, "rb").read().replace(b"\r\n", b""),
     "no bare carriage returns in source")

gate("---" not in body, "no em-dash markup")
gate(not re.search(r"\s--\s", prose_nomac), "no bare -- as punctuation")
gate("robust" not in src.lower(), "no 'robust'")

_t = re.search(r"\\title\{\\bf (.*?)\}\s*\n\s*\n", src, re.S).group(1)
title = _t.replace("\\\\", " ").replace("\n", " ")
gate(len(title.split()) <= 12, "title <= 12 words", len(title.split()))

ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
abw = len(re.sub(r"\\result\{[^}]*\}", "X", ab).split())
gate(abw <= 250, "abstract <= 250 words", abw)

scan = re.sub(r"\\(?:cite[a-z]*|ref|label|includegraphics)\{[^}]*\}", " ", prose_nomac)
# LaTeX lengths are typesetting, not reported values: 0.5em, 1.5em, 0.4pt,
# 0.62\linewidth. Strip them so the gate keeps hunting bare prose decimals.
scan = re.sub(r"\d*\.?\d+\s*(?:em|ex|pt|cm|mm|in|\\linewidth|\\textwidth)", " ", scan)
scan = re.sub(r"\\(?:v|h)space\*?\{[^}]*\}", " ", scan)
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
