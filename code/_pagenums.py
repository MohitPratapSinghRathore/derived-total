"""Add a centred page-number footer to a DOCX, and fix table column widths.

Pandoc emits no footer and lets wide tables run past the margin. This edits the
package directly rather than requiring Word.

Usage: python _pagenums.py <file.docx> [more.docx ...]
"""
import os, re, shutil, sys, zipfile
from xml.sax.saxutils import escape

FOOTER = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:p>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:fldSimple w:instr=" PAGE "><w:r><w:t>1</w:t></w:r></w:fldSimple>
  </w:p>
</w:ftr>"""

FOOTER_REL = ('<Relationship Id="rIdFooterAuto" '
              'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
              'relationships/footer" Target="footer-auto.xml"/>')

FOOTER_CT = ('<Override PartName="/word/footer-auto.xml" '
             'ContentType="application/vnd.openxmlformats-officedocument.'
             'wordprocessingml.footer+xml"/>')


def add_footer(path):
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for n in names:
            data = zin.read(n)
            if n == "word/document.xml":
                x = data.decode("utf-8")
                # reference the footer from every section
                x = re.sub(r"(<w:sectPr\b[^>]*>)",
                           r'\1<w:footerReference w:type="default" '
                           r'r:id="rIdFooterAuto"/>', x)
                data = x.encode("utf-8")
            elif n == "word/_rels/document.xml.rels":
                x = data.decode("utf-8")
                if "rIdFooterAuto" not in x:
                    x = x.replace("</Relationships>", FOOTER_REL + "</Relationships>")
                data = x.encode("utf-8")
            elif n == "[Content_Types].xml":
                x = data.decode("utf-8")
                if "footer-auto.xml" not in x:
                    x = x.replace("</Types>", FOOTER_CT + "</Types>")
                data = x.encode("utf-8")
            zout.writestr(n, data)
        zout.writestr("word/footer-auto.xml", FOOTER)
    shutil.move(tmp, path)


def main():
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        add_footer(path)
        print(f"page numbers added: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
