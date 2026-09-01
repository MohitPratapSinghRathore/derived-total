"""Reorder the results subsections so the argument runs in dependency order.

The sections accumulated in the order the experiments finished, not the order a
reader needs them. In particular the causal intervention, which completes the
belief-consistency argument, sat well after two sections about generality.

Target order:
  mechanism   decay, aggregate fails, action-relevant works, errors are coherent,
              inducing a false belief, how much planning error is memory
  generality  scale, the intervention, the second domain
  method      the degenerate metric
  verdicts    what the predictions came to
"""
import os, re

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()

start = s.index("\\section{Results}")
end = s.index("\\section{Discussion}")
head, body, tail = s[:start], s[start:end], s[end:]

# stale title: the domain did eventually reach a measurable regime
body = body.replace(
    "\\subsection{A second domain that did not work, and why}",
    "\\subsection{A second domain, and what it took to build one}")

parts = re.split(r"(?m)^(\\subsection\{)", body)
# parts[0] is the section head plus any preamble before the first subsection
pre = parts[0]
subs = []
for i in range(1, len(parts), 2):
    chunk = parts[i] + parts[i + 1]
    title = chunk[len("\\subsection{"):chunk.index("}")]
    subs.append((title, chunk))

ORDER = [
    "State and behaviour both decay with depth",
    "Aggregate fidelity does not explain behaviour",
    "Action-relevant fidelity does",
    "The errors are coherent",
    "Inducing a false belief changes the action",
    "How much of the planning error is really memory",
    "Scale",
    "An intervention, evaluated by the diagnostic",
    "A second domain, and what it took to build one",
    "A degenerate metric, declared",
    "What the predictions came to",
]

have = {t for t, _ in subs}
missing = [t for t in ORDER if t not in have]
extra = [t for t in have if t not in ORDER]
assert not missing, f"missing: {missing}"
assert not extra, f"unlisted: {extra}"

by_title = dict(subs)
body = pre + "".join(by_title[t] for t in ORDER)
open(p, "w", encoding="utf-8").write(head + body + tail)
print("reordered:", len(ORDER), "subsections")
