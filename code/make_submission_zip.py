"""Assemble the LaTeX source archive a preprint server can compile.

Preprints.org accepts a Word file or a complete LaTeX source archive. The archive
has to be self-contained, so this collects the manuscript, the generated macro
file, the bibliography, the figures and the title page, then verifies the result
by compiling it in a scratch directory rather than trusting that the file list is
complete. A source archive that does not build is worse than no archive, and the
only way to know is to build it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper_scaling")
BUILD = os.path.join(ROOT, "build_scaling")
TECTONIC = os.path.join(ROOT, "tools", "tectonic.exe")
OUT = os.path.join(BUILD, "ScaleFixesLocalErrorsFirst_LaTeX.zip")

FILES = ["main.tex", "titlepage.tex", "results_macros.tex", "refs.bib"]


def main() -> int:
    missing = [f for f in FILES if not os.path.exists(os.path.join(PAPER, f))]
    if missing:
        print("missing sources:", missing)
        return 1

    figs = []
    fdir = os.path.join(PAPER, "figures")
    if os.path.isdir(fdir):
        figs = [f for f in sorted(os.listdir(fdir))
                if f.lower().endswith((".png", ".pdf", ".jpg"))]

    os.makedirs(BUILD, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(PAPER, f), f)
        for f in figs:
            z.write(os.path.join(fdir, f), os.path.join("figures", f))
    size = os.path.getsize(OUT)
    print(f"wrote {os.path.basename(OUT)}: {len(FILES) + len(figs)} files, "
          f"{size / 1e6:.2f} MB")

    # Verify by compiling the archive's own contents, not the working tree.
    if not os.path.exists(TECTONIC):
        print("tectonic not found; archive written but NOT verified")
        return 0
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(OUT) as z:
            z.extractall(td)
        r = subprocess.run([TECTONIC, "-X", "compile", "main.tex",
                            "--outdir", td],
                           cwd=td, capture_output=True, text=True)
        pdf = os.path.join(td, "main.pdf")
        if r.returncode != 0 or not os.path.exists(pdf):
            print("ARCHIVE DOES NOT COMPILE")
            print(r.stderr[-1500:])
            return 1
        try:
            import pypdf
            n = len(pypdf.PdfReader(pdf).pages)
            txt = "".join((p.extract_text() or "")
                          for p in pypdf.PdfReader(pdf).pages)
            bad = txt.count("??")
            print(f"archive compiles standalone: {n} pages, "
                  f"{bad} unresolved references")
            if bad:
                print("  unresolved references in the archive build")
                return 1
        except ImportError:
            print("archive compiles standalone")
        shutil.copy(pdf, os.path.join(BUILD, "zip_verification.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
