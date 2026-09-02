"""Where does the paper assert something without support?

Reviewers mark uncited claims in related work first. This lists each paragraph
of the section with its citation count, and flags sentences that make a claim
about prior work while carrying no citation.
"""
import os, re

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()
sec = s[s.index(r"\section{Related Work}"):s.index(r"\section{Design Problem")]

CLAIMY = re.compile(
    r"\b(establish|show|demonstrat|report|argue|prove|find that|literature|"
    r"prior work|standard practice|usually|commonly|known to|it is known)\b",
    re.I)

print(f"{'paragraph':44s} {'cites':>5} {'uncited claim-like sentences':>30}")
for para in sec.split(r"\paragraph{")[1:]:
    title = para[:para.index("}")]
    body = para[para.index("}") + 1:]
    n = len(re.findall(r"\\cite[tp]", body))
    flat = " ".join(body.split())
    sentences = re.split(r"(?<=[.])\s+", flat)
    flagged = [x for x in sentences
               if CLAIMY.search(x) and "\\cite" not in x and len(x) > 60]
    print(f"{title[:44]:44s} {n:5d} {len(flagged):30d}")
    for f in flagged:
        print(f"      ! {f[:150]}")
