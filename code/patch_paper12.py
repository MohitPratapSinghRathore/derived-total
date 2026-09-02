"""Add the abstention result: is the measurement useful, or only diagnostic?"""
import os

p = os.path.join(os.path.dirname(__file__), "..", "paper", "main.tex")
s = open(p, encoding="utf-8").read()

SEC = r"""\subsection{Is the measurement useful, or only diagnostic?}
\label{sec:abstain}

Everything so far is a claim about mechanism. A reader is entitled to ask whether
the same measurement is worth anything operationally, and the only baseline that
matters is the model's own confidence. If an external state monitor merely
recovers what the model already knows about its own uncertainty, it is
interesting and redundant.

We test this as selective prediction. A system that wants to avoid acting on a
decayed state abstains on the positions it trusts least, and the question is what
to rank by. We compare the model's top-1 probability, the entropy of its
next-move distribution, and the probe's certainty about the squares the chosen
move touches. That last signal uses only the probe's own output, so it is
computable at inference from activations, the probe having been fitted against an
oracle offline.

One version of this experiment is not admissible and we report it only as a
bound. Ranking by whether the probe is \emph{wrong} on those squares requires the
oracle at inference, which is the very thing an abstention rule exists to avoid.
It is an upper bound on what a perfect state monitor could deliver, not a method.

\input{tab_abstain}

The honest result is modest. Probe certainty alone is a weaker signal than the
model's own confidence, with an AUC of \result{abAucProbe} against
\result{abAucConf}. Combined with confidence it is better than either alone: at
half coverage the illegal-move rate among retained positions falls to
\result{abComb50} against \result{abConf50} for confidence and
\result{abEnt50} for entropy, and in a joint fit both terms carry weight,
\result{abBetaConf} for confidence and \result{abBetaProbe} for probe certainty.
An external monitor therefore adds something the model does not self-report, and
it adds less than we hoped.

The gap to the bound is the interesting part. Ranking by oracle-verified
action-relevant state reaches \result{abOracle50} at the same coverage, less than
half the base rate of \result{abBase}. Most of the benefit an ideal state monitor
would deliver is not captured by the probe's own certainty, which says the
limitation is the quality of the state read-out rather than the premise. Better
read-outs, not a different signal, are the lever.

"""

anchor = r"""\subsection{A horizon that does not depend on behaviour}"""
assert anchor in s
s = s.replace(anchor, SEC + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched: abstention section added")
