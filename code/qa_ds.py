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

# De-escaped control sequences. Writing LaTeX through a shell heredoc turns \t
# into a tab, \r into a carriage return and \e into nothing, so "\texttt" can
# reach the file as TAB + "exttt" and typesets as visible prose. This has now
# happened twice in this manuscript's history, once as "esult{" and once as
# "exttte1g1", and it survives every other gate because the document still
# compiles. Checking for the control characters catches the cause; checking for
# orphaned fragments catches the ones that arrive without one.
gate("\t" not in src and "\r" not in src,
     "no raw tab or carriage return in source",
     f"tabs {src.count(chr(9))}, CRs {src.count(chr(13))}")
ORPHANS = ("esult{", "ef{", "exttt{", "extbf{", "extit{", "egin{", "nd{",
           "ection{", "aragraph{", "esultdef{", "abel{", "ite{", "mph{")
found = []
for frag in ORPHANS:
    for m in re.finditer(re.escape(frag), src):
        before = src[m.start() - 1] if m.start() else " "
        # A legitimate occurrence is preceded by the rest of its command, so the
        # character before is a letter or a backslash.
        if not (before.isalpha() or before == "\\"):
            found.append(frag + " at " + str(m.start()))
gate(not found, "no de-escaped control sequences", found[:5])

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

# Superseded numbers, detected by provenance rather than by a remembered list.
# Each macro records the result file it came from. The corrected classification
# pass replaced the per-run structure artifacts, so any macro still sourced from
# them is superseded by construction and must not appear in the manuscript. This
# replaces a blocklist that needed widening three times, each time after a stale
# number had already reached the PDF.
SUPERSEDED_SOURCE = "attn_*_{structure,natdiv}"
prov = dict(re.findall(r"\\defresult\{([^}]+)\}\{[^}]*\}\s*%\s*(.*)",
                       open(MAC, encoding='utf-8').read()))
# Some macros from those artifacts are properties of the experiment rather than
# of the classifier, and the corrected pass did not change them: how many runs,
# rungs and seeds there are, and the endpoint parameter counts. One is
# deliberately historical: the pre-split geometry exponent, cited as the value
# the split replaced.
PASS_INVARIANT = {"nRuns", "nRungs", "nSeeds", "paramsSmall", "paramsLarge",
                  "expGeometry"}
stale = sorted(k for k in used
               if SUPERSEDED_SOURCE in prov.get(k, '')
               and k not in PASS_INVARIANT)
gate(not stale, "no macro in prose comes from the superseded pass", stale[:8])

pdf = pypdf.PdfReader(PDF)
txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
gate("??" not in txt, "no ?? in PDF", txt.count("??"))
gate("n/a" not in open(MAC, encoding="utf-8").read().split("% AUTO")[0], "macro file sane")
newest = max(os.path.getmtime(f) for f in (SRC, MAC))
gate(os.path.getmtime(PDF) > newest, "PDF newer than sources")
print(f"  [INFO] pages {len(pdf.pages)}, macros used {len(used)} of {len(defined)}")
sys.exit(1 if fails else 0)
