"""Build the deliverables: author PDF, anonymous PDF, title page.

Regenerates results macros and tables first, so the PDF can never be stale with
respect to the measurements.
"""
import os, subprocess, sys, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "paper")
BUILD = os.path.join(ROOT, "build")
TECTONIC = os.path.join(ROOT, "tools", "tectonic.exe")
if not os.path.exists(TECTONIC):
    TECTONIC = shutil.which("tectonic") or "tectonic"


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode})")
    return r.returncode


def main():
    os.makedirs(BUILD, exist_ok=True)
    py = sys.executable
    rc = 0
    rc |= run([py, os.path.join(ROOT, "code", "make_macros.py")])
    rc |= run([py, os.path.join(ROOT, "code", "make_tables.py")])

    # author build
    rc |= run([TECTONIC, "-X", "compile", "main.tex", "--outdir", BUILD], cwd=PAPER)

    # anonymous build: same source, ANON defined, separate jobname
    anon = os.path.join(PAPER, "main_anon.tex")
    open(anon, "w", encoding="utf-8").write("\\def\\ANON{}\\input{main.tex}\n")
    rc |= run([TECTONIC, "-X", "compile", "main_anon.tex", "--outdir", BUILD], cwd=PAPER)

    # title page
    rc |= run([TECTONIC, "-X", "compile", "titlepage.tex", "--outdir", BUILD], cwd=PAPER)

    # optional DOCX if pandoc is present
    if shutil.which("pandoc"):
        for src, out in (("main.tex", "SeparatingStateLoss.docx"),
                         ("titlepage.tex", "TitlePage.docx")):
            run(["pandoc", src, "-o", os.path.join(BUILD, out),
                 "--resource-path=figures"], cwd=PAPER)
    else:
        print("note: pandoc not installed; DOCX deliverables skipped")

    print("\nbuild dir:", BUILD)
    for f in sorted(os.listdir(BUILD)):
        print("  ", f)
    return rc


if __name__ == "__main__":
    sys.exit(main())
