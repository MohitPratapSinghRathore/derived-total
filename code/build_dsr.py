"""Build the admissibility (design-science) paper: macros, figure, PDF, anonymous PDF, DOCX.

Regenerates the macros and the figure first, so neither the PDF nor the DOCX can
be stale with respect to the result files, then runs the QA gates.

The DOCX needs one extra step. Pandoc does not evaluate \\result{} or \\ifanon, so
both are resolved into a flattened copy before conversion; otherwise the DOCX
would silently contain macro names or both author blocks.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "paper_dsr")
BUILD = os.path.join(ROOT, "build_dsr")
TECTONIC = os.path.join(ROOT, "tools", "tectonic.exe")
PY = sys.executable


def run(cmd, cwd=None, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=quiet, text=True)
    if r.returncode != 0:
        print("FAILED:", " ".join(str(c) for c in cmd))
        if quiet:
            print((r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
    return r.returncode


def flatten_for_docx(anon=False):
    src = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
    mac = open(os.path.join(PAPER, "results_macros.tex"), encoding="utf-8").read()
    vals = dict(re.findall(r"\\defresult\{([^}]+)\}\{(.*?)\}   %", mac))
    src = re.sub(r"\\input\{results_macros\.tex\}[^\n]*", "", src)

    def res(m):
        return vals.get(m.group(1), "??")

    src = re.sub(r"\\result\{([^}]+)\}", res, src)
    # resolve the anonymity toggle: keep exactly one branch
    keep = 1 if anon else 2

    def toggle(m):
        return m.group(keep)

    # strip the toggle's declaration FIRST: it contains "\\ifanon", and a toggle
    # regex run before this line is removed matches from the declaration onward
    src = src.replace("\\newif\\ifanon\n\\ifdefined\\ANON\\anontrue\\else\\anonfalse\\fi\n", "")
    src = re.sub(r"\\ifanon(.*?)\\else(.*?)\\fi", toggle, src, flags=re.S)
    out = os.path.join(PAPER, "main_docx_anon.tex" if anon else "main_docx.tex")
    open(out, "w", encoding="utf-8").write(src)
    return out, src.count("??")


def main():
    os.makedirs(BUILD, exist_ok=True)
    rc = 0
    rc |= run([PY, os.path.join(ROOT, "code", "make_macros_dsr.py")])
    rc |= run([PY, os.path.join(ROOT, "code", "figures_dsr.py")])
    rc |= run([TECTONIC, "-X", "compile", "main.tex", "--outdir", BUILD],
              cwd=PAPER, quiet=True)
    anon = os.path.join(PAPER, "main_anon.tex")
    open(anon, "w", encoding="utf-8").write("\\def\\ANON{}\\input{main.tex}\n")
    rc |= run([TECTONIC, "-X", "compile", "main_anon.tex", "--outdir", BUILD],
              cwd=PAPER, quiet=True)

    if shutil.which("pandoc"):
        for is_anon, name in ((False, "InadmissibleScalingForecasts.docx"),
                              (True, "InadmissibleScalingForecasts_anon.docx")):
            flat, unresolved = flatten_for_docx(is_anon)
            if unresolved:
                print(f"  WARNING: {unresolved} unresolved macros in {flat}")
            rc |= run(["pandoc", os.path.basename(flat), "--citeproc",
                       "--bibliography=refs.bib", "--resource-path=figures",
                       "-o", os.path.join(BUILD, name)], cwd=PAPER, quiet=True)
    rc |= run([PY, os.path.join(ROOT, "code", "qa_dsr.py")])

    # anonymous build must not carry author identity
    try:
        import fitz
        t = "".join(p.get_text() for p in fitz.open(os.path.join(BUILD, "main_anon.pdf")))
        leaks = [n for n in ("Rathore", "Kalsi", "Meduri", "Oviqo") if n in t.split("References")[0]]
        print(f"  [{'PASS' if not leaks else 'FAIL'}] anonymous PDF body has no "
              f"identity leaks {leaks}")
        rc |= int(bool(leaks))
    except ImportError:
        pass

    print("\nbuild_dsr:")
    for f in sorted(os.listdir(BUILD)):
        if f.endswith((".pdf", ".docx")):
            print("  ", f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
