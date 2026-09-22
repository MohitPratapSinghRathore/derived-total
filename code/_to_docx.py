"""Pre-resolve LaTeX that pandoc cannot handle, then emit a DOCX-ready file.

Pandoc does not run natbib, does not evaluate the \\ifanon toggle, and does not
know what \\result{} means. This resolves all of that first, so the DOCX carries
the same numbers and citations as the PDF rather than a degraded version.

Usage:  python _to_docx.py full   |   python _to_docx.py anon
Writes: paper/main_docx.tex
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
BUILD = os.path.join(HERE, "..", "build")
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
ANON = MODE == "anon"


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------- inputs
src = read(os.path.join(PAPER, "main.tex"))

# expand \input{...} for the generated pieces, keeping order
def expand(m):
    name = m.group(1)
    for cand in (name, name + ".tex"):
        p = os.path.join(PAPER, cand)
        if os.path.exists(p):
            return read(p)
    return ""


src = re.sub(r"\\input\{([^}]*)\}", expand, src)

# ------------------------------------------------------- \result{} macros
macros = {}
for m in re.finditer(r"\\defresult\{([^}]*)\}\{([^}]*)\}", src):
    macros[m.group(1)] = m.group(2)
src = re.sub(r"\\defresult\{[^}]*\}\{[^}]*\}[^\n]*\n", "", src)
src = re.sub(r"\\result\{([^}]*)\}", lambda m: macros.get(m.group(1), "??"), src)
# the macro definitions themselves are no longer needed
src = re.sub(r"\\makeatletter.*?\\makeatother", "", src, flags=re.S)

# --------------------------------------------------------- the anon toggle
def resolve_toggle(text):
    out, i = [], 0
    pat = re.compile(r"\\ifanon")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        depth, j = 1, m.end()
        else_at = None
        while depth and j < len(text):
            if text.startswith(r"\ifanon", j) or text.startswith(r"\ifdefined", j):
                depth += 1; j += 7
            elif text.startswith(r"\else", j) and depth == 1:
                else_at = j; j += 5
            elif text.startswith(r"\fi", j):
                depth -= 1
                if depth == 0:
                    break
                j += 3
            else:
                j += 1
        body = text[m.end():j]
        if else_at is not None:
            a = text[m.end():else_at]
            b = text[else_at + 5:j]
        else:
            a, b = body, ""
        out.append(a if ANON else b)
        i = j + 3
    return "".join(out)


src = src.replace(r"\ifdefined\ANON\anontrue\else\anonfalse\fi", "")
src = re.sub(r"\\newif\\ifanon", "", src)
src = resolve_toggle(src)

# ------------------------------------------------- \ref and \eqref from .aux
labels = {}
aux = os.path.join(BUILD, "main.aux")
if os.path.exists(aux):
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", read(aux)):
        labels[m.group(1)] = m.group(2)
src = re.sub(r"\\(?:eq)?ref\{([^}]*)\}",
             lambda m: labels.get(m.group(1), "??"), src)

# --------------------------------------------- \citet and \citep from .bbl
entries = {}
bbl = os.path.join(BUILD, "main.bbl")
if os.path.exists(bbl):
    txt = read(bbl)
    # apalike writes \bibitem[Author and Author, 2026]{key}
    for m in re.finditer(r"\\bibitem\[([^\]]*)\]\{([^}]*)\}", txt):
        tag, key = m.group(1), m.group(2)
        tag = re.sub(r"\\protect\s*|\{|\}", "", tag).strip()
        mm = re.match(r"(.*),\s*(\d{4}[a-z]?)\s*$", tag)
        if mm:
            entries[key] = (mm.group(1).strip(), mm.group(2))
        else:
            mm = re.match(r"(.*?)\((\d{4}[a-z]?)\)", tag)
            entries[key] = ((mm.group(1).strip(), mm.group(2)) if mm
                            else (tag, ""))


def cite(m):
    kind, keys = m.group(1), [k.strip() for k in m.group(2).split(",")]
    parts = []
    for k in keys:
        a, y = entries.get(k, (k, ""))
        parts.append(f"{a} {y}".strip() if kind in ("citep", "citealp")
                     else f"{a} ({y})")
    if kind in ("citep", "citealp"):
        return "(" + "; ".join(parts) + ")"
    return ", ".join(parts)


src = re.sub(r"\\(citet|citep|citealp|citealt)\{([^}]*)\}", cite, src)

# ------------------------------------------------- description lists to bold
def desc(m):
    body = m.group(1)
    body = re.sub(r"\\item\[([^\]]*)\]", lambda x: "\n\n\\textbf{" + x.group(1) + "} ",
                  body)
    return body


src = re.sub(r"\\begin\{description\}(?:\[[^\]]*\])?(.*?)\\end\{description\}",
             desc, src, flags=re.S)

# ------------------------------------------------------------- figures
# pandoc resolves the path relative to its working directory and does not read
# \graphicspath, so the directory is written into every path here. Without this
# the DOCX comes out with captions and no images.
def _fig(m):
    name = m.group(1)
    if not name.startswith("figures/"):
        name = "figures/" + name
    if not os.path.splitext(name)[1]:
        name += ".png"
    return "\\includegraphics[width=6in]{%s}" % name


src = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", _fig, src)
src = src.replace(r"\graphicspath{{figures/}}", "")

# ------------------------------------------------- bibliography as paragraphs
if os.path.exists(bbl):
    txt = read(bbl)
    items = re.split(r"\\bibitem\[[^\]]*\]\{[^}]*\}", txt)[1:]
    items = [i for i in items if i.strip()]
    refs = []
    for it in items:
        it = it.split(r"\end{thebibliography}")[0]
        it = re.sub(r"\\newblock\s*", " ", it)
        it = re.sub(r"\s+", " ", it).strip()
        if it:
            refs.append(it)
    refs.sort(key=lambda s: s.lower())
    bib = "\\section*{References}\n\n" + "\n\n".join(refs) + "\n"
else:
    bib = ""
src = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}",
             lambda _m: bib, src)

# ------------------------------------------- numbering that pandoc will not do
# Sections and subsections are numbered in document order, and every table and
# figure caption is prefixed with its number, so the DOCX can be read on its own
# and every cross-reference in it resolves.
def _number_sections(text):
    out, counters = [], [0, 0]
    for line in text.split("\n"):
        m = re.match(r"\\section\{(.*)\}\s*$", line)
        if m and "*" not in line:
            counters[0] += 1
            counters[1] = 0
            out.append("\\section{%d. %s}" % (counters[0], m.group(1)))
            continue
        m = re.match(r"\\subsection\{(.*)\}\s*$", line)
        if m and "*" not in line:
            counters[1] += 1
            out.append("\\subsection{%d.%d. %s}"
                       % (counters[0], counters[1], m.group(1)))
            continue
        out.append(line)
    return "\n".join(out)


def _number_floats(text):
    """Prefix each caption with Table N. / Figure N., in document order."""
    counts = {"table": 0, "figure": 0}
    out, i = [], 0
    pat = re.compile(r"\\begin\{(table|figure|longtable)\}")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        kind = "table" if m.group(1) in ("table", "longtable") else "figure"
        end = text.find("\\end{%s}" % m.group(1), m.end())
        end = len(text) if end < 0 else end
        body = text[m.start():end]
        counts[kind] += 1
        label = "Table" if kind == "table" else "Figure"
        prefix = "\\caption{%s %d. " % (label, counts[kind])
        body = re.sub(r"\\caption\{", lambda _m: prefix, body, count=1)
        out.append(text[i:m.start()])
        out.append(body)
        i = end
    return "".join(out)


def _flatten_headers(text):
    r"""pandoc drops \multicolumn, which leaves a short header row over a wide
    table. The spanning row is folded into the header beneath it instead."""
    def one(m):
        span, head = m.group(1), m.group(2)
        groups = re.findall(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{([^}]*)\}", span)
        if not groups:
            return m.group(0)
        names = []
        for n, title in groups:
            names.extend([title.strip()] * int(n))
        cells = [c.strip() for c in head.split("&")]
        lead = len(cells) - len(names)
        merged = cells[:max(lead, 0)] + [
            (f"{names[k]}: {cells[lead + k]}" if lead + k < len(cells) else names[k])
            for k in range(len(names))]
        return "\\midrule\n" + " & ".join(merged) + " \\\\\n"

    return re.sub(r"(?:\\midrule\n)?([^\n]*\\multicolumn[^\n]*)\\\\\n"
                  r"(?:\\cmidrule[^\n]*\n)?([^\n]*)\\\\\n",
                  one, text)


src = _number_sections(src)
src = _number_floats(src)
src = _flatten_headers(src)

# ------------------------------------------------------------- tidy up
src = src.replace(r"\clearpage", "").replace(r"\newpage", "")
src = re.sub(r"\\label\{[^}]*\}", "", src)
src = re.sub(r"\\hypersetup\{[^}]*\}", "", src)
src = re.sub(r"\n{3,}", "\n\n", src)

out = os.path.join(PAPER, "main_docx.tex")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(src)
print(f"wrote {out}  ({MODE}; {len(macros)} result macros, {len(labels)} labels, "
      f"{len(entries)} citations resolved)")
